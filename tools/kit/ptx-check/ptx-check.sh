#!/bin/bash
# tools/ptx-check.sh (ENGINEER-3, orchestrator 23:3x; new file under the frozen kit, VERIFY re-read requested). The PTX byte-copy
# check the captains did by hand before every load (orchestrator RULE 22:42:47), as one command.
# Usage: bash .local/tools/ptx-check.sh <candidate payload dir> <head payload dir>
#   Reads <dir>/flash.ptx (required) and, when present, the cubin of the same build: <dir>/flash.cubin, else the sibling
#   cubin-<name>/flash.cubin (payload-X -> cubin-X), for registers per entry (cuobjdump is not on the Mac; the count is read from
#   the cubin ELF the way ENGINEER-5's cubinregs.rb does: EIATTR_REGCOUNT in .nv.info; no counts found = not compared).
# Per .entry (and .func, if any) body of flash.ptx: the st.local.b8 count (byte-wise stores to local memory: the byte-copy
# loop that ARGS-DIRECT removed and 450de6efb reintroduced in resident_launch_mixed).
# FAIL (exit 1): a common entry has MORE st.local.b8 than the head; a NEW entry has any st.local.b8; a head entry is MISSING.
# Listed, not a failure: NEW entries without st.local.b8; FEWER st.local.b8 than the head (a byte copy removed); REG changes.
# Input errors, exit 3 (fail closed): a missing or empty flash.ptx, a file that ends inside a declaration, body or block comment,
# a duplicate definition, an .entry or .func that cannot be classified, text after a closing brace, no .entry at all:
# "PTX CHECK <rev> FAIL input: <reason>". Every .entry and .func is parsed, qualified or not; prototypes (";" before "{") have no
# body; braces and st.local.b8 inside comments and strings do not count (VERIFY FIX FIRST 23:55:28, fixed 00:0x).
# Output: one line per difference, then "PTX CHECK <rev> clean" (exit 0) or "PTX CHECK <rev> FAIL <entry>[,<entry>...]" (exit 1).
# Cost: reads two PTX files (about 7.6 MB each) and two cubins, a few seconds. Gain: the hand diff before every T load.
set -u
[ $# = 2 ] || { echo "usage: ptx-check.sh <candidate payload dir> <head payload dir>" >&2; exit 2; }
exec ruby -e '
cand, head = ARGV
def rev_of(d) b = File.read(File.join(d, "BUILD-REVISION")).strip rescue ""; b =~ /\A[0-9a-f]{9,40}\z/ ? b[0, 9] : File.basename(d).sub(/\Apayload-/, "") end
def entries(path)
  raise "missing #{path}" unless File.file?(path) && File.size(path) > 0
  out = {}; state = :out; decl = ""; dline = 0; name = nil; depth = 0; inblk = false; ln = 0
  strip = lambda do |l|
    return l if !inblk && !l.include?("/") && !l.include?(34.chr)
    s = ""; i = 0
    while i < l.size
      if inblk then j = l.index("*/", i); if j then inblk = false; i = j + 2 else i = l.size end; next end
      c = l[i]
      if c == "/" && l[i + 1] == "/" then break
      elsif c == "/" && l[i + 1] == "*" then inblk = true; i += 2; next
      elsif c == 34.chr then j = l.index(34.chr, i + 1) or raise "unterminated string at line #{ln}"; s << " "; i = j + 1; next
      end
      s << c; i += 1
    end
    s
  end
  File.foreach(path) do |raw|
    ln += 1; t = strip.call(raw.chomp)
    case state
    when :out
      next unless t =~ /(?:\A|\s)\.(entry|func)\b/
      state = :decl; decl = ""; dline = ln
    end
    if state == :decl
      decl << t << " "
      pd = 0; term = nil
      decl.each_char.with_index { |ch, k| pd += 1 if ch == "("; pd -= 1 if ch == ")"; if pd == 0 && (ch == "{" || ch == ";") then term = k; break end }
      next unless term
      head = decl[0...term]
      m = head.match(/\A\s*(?:\.(?:visible|weak|extern)\s+)*\.(entry|func)\s*(?:\((?:[^()]|\([^()]*\))*\)\s*)?([A-Za-z_$][\w$]*)\s*(?:\((?:[^()]|\([^()]*\))*\))?(?:\s*\.(?:maxntid|reqntid|minnctapersm|maxnreg|maxclusterrank|reqnctapercluster|explicitcluster|noreturn|pragma)\b[^{;]*)*\s*\z/m)
      raise "unclassified declaration at line #{dline}: #{head.strip[0, 80]}" unless m
      key = (m[1] == "func" ? "func:" : "") + m[2]
      if decl[term] == ";" then state = :out; rest = decl[term + 1..].strip; raise "text after a prototype at line #{ln}" unless rest.empty?; next end
      raise "duplicate definition #{key} at line #{dline}" if out.key?(key)
      name = key; out[name] = 0; depth = 0; state = :body; t = decl[term..]
    end
    if state == :body && !t.include?("{") && !t.include?("}")
      out[name] += t.scan(/\bst\.local\.b8\b/).size if t.include?("st.local.b8")
      next
    end
    if state == :body
      t.each_char.with_index do |ch, k|
        if ch == "{" then depth += 1
        elsif ch == "}"
          depth -= 1
          if depth == 0
            body_part = t[0..k]; out[name] += body_part.scan(/\bst\.local\.b8\b/).size
            raise "text after the closing brace of #{name} at line #{ln}" unless t[k + 1..].strip.empty?
            state = :out; name = nil; break
          end
        end
      end
      out[name] += t.scan(/\bst\.local\.b8\b/).size if state == :body
    end
  end
  raise "incomplete: the file ends inside #{state == :body ? "the body of #{name}" : "a declaration (line #{dline})"}" unless state == :out
  raise "incomplete: the file ends inside a block comment" if inblk
  raise "no .entry in #{path}" if out.keys.none? { |k| !k.start_with?("func:") }
  out
end
def cubin_of(d) [File.join(d, "flash.cubin"), File.join(File.dirname(d), File.basename(d).sub(/\Apayload-/, "cubin-"), "flash.cubin")].find { |p| File.file?(p) } end
def regs(path)
  # EIATTR_REGCOUNT (attribute 0x2f, format EIFMT_SVAL 4: symbol index, registers) in the .nv.info sections, symbol names from
  # .symtab (the method of ENGINEER-5 cubinregs.rb that gives 154 for resident_launch_mixed; sh_info here is a symbol index).
  return nil unless path
  b = File.binread(path); return nil unless b[0, 4] == "\x7fELF".b
  shoff = b[0x28, 8].unpack1("Q<"); shentsize, shnum, shstrndx = b[0x3a, 6].unpack("S<3")
  hs = (0...shnum).map { |i| b[shoff + i * shentsize, 64].unpack("L<L<Q<Q<Q<Q<L<L<Q<Q<") }
  st = hs[shstrndx]; names = b[st[4], st[5]]; nm = ->(h) { names[h[0]..][/\A[^\0]*/] }
  sym = hs.find { |h| nm[h] == ".symtab" } or return nil
  strt = hs[sym[6]]; strs = b[strt[4], strt[5]]
  syms = (0...sym[5] / 24).map { |i| strs[b[sym[4] + i * 24, 4].unpack1("L<")..][/\A[^\0]*/] }
  r = {}
  hs.each do |h|
    next unless nm[h].start_with?(".nv.info")
    d = b[h[4], h[5]]; i = 0
    while i + 4 <= d.size
      fmt, attr = d[i, 2].unpack("CC")
      if fmt == 4 then size = d[i + 2, 2].unpack1("S<"); data = d[i + 4, size]; r[syms[data[0, 4].unpack1("L<")]] = data[4, 4].unpack1("L<") if attr == 0x2f && size >= 8; i += 4 + size
      else i += 4 end
    end
  end
  r.empty? ? nil : r
rescue StandardError
  nil
end
rev = rev_of(cand)
begin
  c = entries(File.join(cand, "flash.ptx")); h = entries(File.join(head, "flash.ptx"))
rescue StandardError => e
  puts "PTX CHECK #{rev} FAIL input: #{e.message}"; exit 3
end
fails = []
new_e = c.keys - h.keys; gone = h.keys - c.keys
ne = ->(x) { x.keys.count { |k| !k.start_with?("func:") } }; puts "PTX DEFINITIONS head #{ne[h]} .entry + #{h.size - ne[h]} .func, candidate #{ne[c]} .entry + #{c.size - ne[c]} .func: #{new_e.size} new, #{gone.size} missing; st.local.b8 total head #{h.values.sum} candidate #{c.values.sum}"
gone.sort.each { |e| puts "MISSING #{e} (head st.local.b8 #{h[e]})"; fails << e }
new_e.sort.each { |e| if c[e] > 0 then puts "NEW #{e} st.local.b8 #{c[e]} (byte copy in a new entry)"; fails << e else puts "NEW #{e} st.local.b8 0" end }
(c.keys & h.keys).sort.each do |e|
  next if c[e] == h[e]
  if c[e] > h[e] then puts "BYTECOPY #{e} st.local.b8 head #{h[e]} -> candidate #{c[e]}"; fails << e
  else puts "FEWER #{e} st.local.b8 head #{h[e]} -> candidate #{c[e]} (not a failure)" end
end
rc, rh = regs(cubin_of(cand)), regs(cubin_of(head))
if rc && rh
  (rc.keys & rh.keys).sort.each { |e| puts "REG #{e} head #{rh[e]} -> candidate #{rc[e]} (registers, not a failure)" if rc[e] != rh[e] }
  puts "REGISTERS read from both cubins (#{rh.size} and #{rc.size} kernels)"
else
  puts "REGISTERS not compared: cubin #{rc ? "" : "of the candidate "}#{rh ? "" : "of the head "}not found or not ELF"
end
if fails.empty? then puts "PTX CHECK #{rev} clean"; exit 0
else puts "PTX CHECK #{rev} FAIL #{fails.uniq.join(",")}"; exit 1 end
' "$1" "$2"
