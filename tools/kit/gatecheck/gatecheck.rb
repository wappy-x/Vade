# Lane M gatecheck.rb v4 (22:1x: captures rule, FORM, strict integers; v3 kept as tools/gatecheck-v3.rb). v2 (2026-11-06 21:2x; fixes lane VERIFY's audits notes/reviews/verify-gate-tooling-2026-11-06.md and
# notes/reviews/verify-ladder3-2026-11-06.md; v1 kept as tools/gatecheck-v1.rb). ONE verdict engine for gate-hook-ladder3.sh
# and gate-hook4.sh. FAIL CLOSED: a missing file, a missing or failed completion marker, a malformed, duplicate, non-finite or
# out-of-range number, a missing session or rank, a missing reference file, a missing or out-of-order rung, any FAILED in a host
# log, or any internal error is a FAIL/VOID/BLOCKED/ERROR, never a PASS. Exit: 0 every rung PASS; 1 any FAIL/VOID/BLOCKED;
# 2 no FAIL but a host gate PENDING; 3 internal or SPEC error. The last line is always "GATECHECK END rungs=<n> exit=<code>"
# (the hooks require it, and one VERDICT line per rung, and the HIGHEST PASSING RUNG line).
#
# Usage: ruby gatecheck.rb SPEC.  SPEC lines (space separated, # comments):
#   PAIR <rank0 box> <rank1 box>
#   FORM <head loads> <loads per rung>   (ladder: FORM 1 1; H,T,T: FORM 1 2; two heads: FORM 2 n)
#   HEAD <evidence load dir>...      (each .../<rev folder>, e.g. evidence/m-glm-lad3-h1/bc8a92ea0-native)
#   HEADREV <full 40-hex sha>
#   RUNG <T1|T2|...> <label> <full sha> <host gates log> <load dir>[,<load dir>...] [counter=1] [bf16=<bytes>:<matrices>]
# Rungs must be exactly T1, T2, ..., Tn in this order (no gap, no duplicate); each must descend from the head and the rung below.
#
# Checks:
#  inventory  every box folder; BUILD-REVISION == full sha; glm.exit == 0; every job folder COMPLETE == RESIDENT DECODE SUITE PASS;
#             every run 1-3 of every job on BOTH ranks: COMPLETE == RESIDENT DECODE RUN PASS, COUNTS, TIMING, PREFILL_TIMING,
#             PROJECTION_DISPATCH, INPUT_IDS, OUTPUT_IDS, PROMPT_LOGITS and logits 0..N-1 (15: 64, 127: 64, 1,040: 32,
#             4,158: 48) present and non-empty, no logits N; every metric field of every run on both ranks parsed strictly.
#  numbers    a field is "key=value" bounded by whitespace, appears exactly once, and the whole value parses (counts as integers
#             >= 0, rates and seconds as finite > 0); anything else FAILS.
#  identity   rank from the log (the rank 0 box prints only rank=0 TOKEN lines); start and end ISO times from each rank's log
#             (first and last ISO line); end after start; the two ranks within 120 s of each other; no time in the future.
#  bytes      SHA-256 of every byte file: rank 1 = rank 0, runs 2-3 = run 1, every rung load = the head; the head = the golden
#             reference (every run 1-3 of 15, 127, 1,040 must hold the exact inventory) and the lane T long reference (17 logit
#             hashes, OUTPUT_IDS first 17, PROMPT_LOGITS).
#  decode     rank 0, named runs (15/127: 1-3; 1,040/4,158: 2-3); loss beyond 0.7/0.5/1.0/2.0 percent FAILS; head spread > 4% RERUN.
#  prompt     rank 0 runs 1-3 at 1,040 and 4,158; one head load: allowance 1.0 / 1.5 percent; several: larger of 1 and half the
#             head spread, cap 2; the head's own session spread over 4 percent RERUN.
#  steps      sessions 2-3 at 1,040 (31 steps) and 4,158 (47); train spikes over head spikes + SPIKE_MARGIN (5) FAIL.
#  captures   v4 (lane VERIFY's notes/reviews/verify-captures-rule-2026-11-06.md, tools/captures_rule.rb): raw graph_captures of
#             sessions 2 and 3 (session 1 excluded), rank 0, every length, decode_steps exactly 63/63/31/47; H = all head counts,
#             spread = max(H)-min(H); each candidate load separately: both sessions <= max(H)+spread and n*(t2+t3) <= sum(H)+n*spread.
#  form       a "FORM <head loads 1|2> <loads per rung 1|2>" line is required and must match the HEAD and RUNG lines; every
#             evidence folder named once. Integer fields must match [0-9]+ exactly.
#  bf16       per-step project_bf16_ordered/project_f32_ordered at 15 and RESIDENT READY (both ranks) equal to the head, or
#             READY equal to the rung's bf16=<bytes>:<matrices>; an old head without the fields needs bf16= else FAIL.
#  counters   COUNTS of every load: host_launches equal to the head's; replays, captures and waits reported only.
#  boundary   residual_begin calls / decode_steps for every decode session: equal to the head's.
#  repeats    counter=1: every decode RESIDENT TOKEN line on rank 0 carries each table field exactly once; rates printed.
#  temps      TEMPS.txt before and after, both boxes: a difference of 5 C or more from the head VOIDS.
#  age        train load start minus head load end (ISO times from the logs, both ranks): 0 to 3,600 s, else VOID.
#  host       no "TRAIN HOST FAILED", "GATE FAIL", "MAC CHECKS FAILED" or "STOPPED" line anywhere; the LAST "TRAIN HOST" line is
#             exactly "TRAIN HOST PASS <short sha>"; "PAYLOAD READY <short>" and "MAC BUILD AND CHECKS PASS <short>" before it;
#             no TRAIN HOST line yet = PENDING.
#  ladder     a rung passes only if every lower rung passed (BLOCKED otherwise).
require 'digest'; require 'time'; require_relative 'captures_rule'
STEPS_EXPECTED={"glm-tokens64"=>63,"out127"=>63,"out1040"=>31,"outlong"=>47}
JOBS={"glm-tokens64"=>["15",64],"out127"=>["127",64],"out1040"=>["1,040",32],"outlong"=>["4,158",48]}
FLOOR={"glm-tokens64"=>0.7,"out127"=>0.5,"out1040"=>1.0,"outlong"=>2.0}
DECODE_RUNS={"glm-tokens64"=>[1,2,3],"out127"=>[1,2,3],"out1040"=>[2,3],"outlong"=>[2,3]}
RUNFILES=%w[COMPLETE COUNTS TIMING PREFILL_TIMING PROJECTION_DISPATCH INPUT_IDS OUTPUT_IDS PROMPT_LOGITS]
# Private paths: the environment first, else a "KEY value" line in the file named by VADE_KIT_HOSTS (tools/kit/hosts.example).
def kit_path(key) v=ENV[key]; return v if v && !v.empty?; f=ENV["VADE_KIT_HOSTS"]; return nil unless f && File.file?(f)
  File.foreach(f){|l| k,val=l.sub(/#.*/,"").split(" ",2); return val.strip if k==key && val && !val.strip.empty?}; nil end
GOLDEN=kit_path("GOLDEN")
LONGREF=kit_path("LONGREF")   # base-long17/run1 + logits.sha256
REPO=kit_path("REPO")
$n_rungs=0
def finish(code) puts "GATECHECK END rungs=#{$n_rungs} exit=#{code}"; exit code end
missing=%w[GOLDEN LONGREF REPO].select{|k| Object.const_get(k).nil?}
(puts "GATECHECK ERROR set #{missing.join(", ")} in the environment or in the file named by VADE_KIT_HOSTS"; finish(3)) if missing.any?
class Bad<StandardError; end
def med(v) s=v.sort;n=s.size;raise Bad,"median of nothing" if n==0;n.odd? ? s[n/2] : (s[n/2-1]+s[n/2])/2.0 end
def need(c,msg) raise Bad,msg unless c end
def readf(p) need(File.file?(p) && File.size(p)>0,"missing or empty #{p}");File.binread(p) end
# kind: :int (whole value an integer >= 0) or :pos (whole value a finite float > 0)
def num(text,key,p,kind=:pos)
  m=text.scan(/(?:\A|\s)#{Regexp.escape(key)}=(\S*)/);need(m.size==1,"#{key} appears #{m.size} times in #{p}");v=m[0][0]
  f=kind==:int ? (v=~/\A[0-9]+\z/ ? Integer(v,10) : nil) : (Float(v) rescue nil)
  need(f,"#{key}=#{v[0,30]} does not parse in #{p}");need(kind==:int || f.finite?,"#{key} not finite in #{p}")
  need(kind==:pos ? f>0 : f>=0,"#{key}=#{v} out of range in #{p}");f
end
def sha(p) need(File.file?(p) && File.size(p)>0,"missing or empty #{p}");Digest::SHA256.file(p).hexdigest end
def bytefiles(n) ["INPUT_IDS","OUTPUT_IDS","PROMPT_LOGITS"]+(0...n).map{|i|"#{i}-logits.f32"} end
def iso_times(log,where) t=log.scan(/^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[-+]\d\d:\d\d)\s*$/).flatten;need(t.size>=2,"#{where}: no start/end ISO times in the log")
  a=Time.iso8601(t.first).to_f;b=Time.iso8601(t.last).to_f;need(b>a,"#{where}: end not after start");need(b<=Time.now.to_f+60,"#{where}: end time in the future");[a,b] end
spec=ARGV[0];form=nil;pair=nil;heads=[];headrev=nil;rungs=[];spike_margin=nil
begin
  spike_margin=Integer(ENV["SPIKE_MARGIN"]||"5",10)
  need(spec && File.file?(spec),"no SPEC file")
  File.readlines(spec).each{|l| w=l.sub(/#.*/,"").split;next if w.empty?
    case w[0] when "PAIR" then need(pair.nil? && w.size==3,"bad PAIR line");pair=w[1,2]
    when "FORM" then need(form.nil? && w.size==3 && w[1]=~/\A[12]\z/ && w[2]=~/\A[12]\z/,"bad FORM line (FORM <head loads 1|2> <loads per rung 1|2>)");form=[w[1].to_i,w[2].to_i]
    when "HEAD" then heads+=w[1..] when "HEADREV" then need(headrev.nil?,"two HEADREV lines");headrev=w[1]
    when "RUNG" then need(w.size>=6,"short RUNG line: #{l.strip}");opts=w[6..].to_h{|x|k,v=x.split("=",2);need(%w[counter bf16].include?(k) && v,"bad rung option #{x}");[k,v]}
      rungs<<{name:w[1],label:w[2],rev:w[3],host:w[4],loads:w[5].split(","),opts:opts}
    else raise Bad,"bad SPEC line: #{l.strip}" end}
  need(pair && pair.uniq.size==2,"no valid PAIR");need(heads.any?,"no HEAD");need(headrev=~/\A\h{40}\z/,"HEADREV not a full sha");need(rungs.any?,"no RUNG")
  rungs.each_with_index{|r,i| need(r[:name]=="T#{i+1}","rung #{i+1} is named #{r[:name]}: rungs must be T1..Tn in order, no gap, no duplicate")
    need(r[:rev]=~/\A\h{40}\z/,"rung #{r[:name]} rev not a full sha");need(r[:loads].any? && r[:loads].uniq.size==r[:loads].size,"rung #{r[:name]} loads empty or repeated")}
  need(form,"no FORM line: the gate must declare its head loads and loads per rung");need(heads.size==form[0],"FORM declares #{form[0]} head load(s), SPEC names #{heads.size}")
  rungs.each{|r| need(r[:loads].size==form[1],"FORM declares #{form[1]} load(s) per rung, rung #{r[:name]} names #{r[:loads].size}")}
  allf=(heads+rungs.flat_map{|r|r[:loads]}).map{|d|File.expand_path(d)};need(allf.uniq.size==allf.size,"an evidence folder is named twice (distinct loads must be distinct folders)")
  $n_rungs=rungs.size
rescue StandardError=>e then puts "GATECHECK ERROR #{e.message}";finish(3) end

def load_info(dir,rev,pair,head=false)
  i={dir:dir,boxes:{}}
  pair.each_with_index{|b,r| d="#{dir}/#{b}";need(File.directory?(d),"missing box folder #{d}")
    need(readf("#{d}/BUILD-REVISION").strip==rev,"BUILD-REVISION in #{d} is not #{rev}")
    need(readf("#{d}/glm.exit").strip=="0","glm.exit not 0 in #{d}")
    log=readf("#{d}/glm-run.log").force_encoding("binary")
    ranks=log.scan(/^RESIDENT TOKEN rank=(\d+) /).flatten.uniq;need(ranks==[r.to_s],"rank identity: #{d} logs ranks #{ranks.inspect}, expected #{r}")
    i[:boxes][b]={dir:d,log:log,rank:r,times:iso_times(log,d)}
    JOBS.each{|j,(_,n)| need(readf("#{d}/#{j}/COMPLETE").strip=="RESIDENT DECODE SUITE PASS","#{d}/#{j}/COMPLETE is not RESIDENT DECODE SUITE PASS")
      (1..3).each{|k| rd="#{d}/#{j}/run#{k}";need(File.directory?(rd),"missing #{rd}")
        need(readf("#{rd}/COMPLETE").strip=="RESIDENT DECODE RUN PASS","#{rd}/COMPLETE is not RESIDENT DECODE RUN PASS")
        (RUNFILES+bytefiles(n)).each{|f|readf("#{rd}/#{f}")};need(!File.exist?("#{rd}/#{n}-logits.f32"),"extra logits file #{n} in #{rd}")
        t=readf("#{rd}/TIMING");need(num(t,"decode_steps","#{rd}/TIMING",:int)>0,"decode_steps 0 in #{rd}");num(t,"decode_tokens_per_second","#{rd}/TIMING",:pos)
        num(readf("#{rd}/PREFILL_TIMING"),"prompt_seconds","#{rd}/PREFILL_TIMING",:pos)
        c=readf("#{rd}/COUNTS");%w[graph_replays graph_captures stream_waits host_launches].each{|f|num(c,f,"#{rd}/COUNTS",:int)}
        pd=readf("#{rd}/PROJECTION_DISPATCH");need(num(pd,"decode_steps","#{rd}/PROJECTION_DISPATCH",:int)>0,"PROJECTION_DISPATCH decode_steps 0 in #{rd}")
        %w[project_bf16_ordered project_f32_ordered].each{|f| num(pd,f,"#{rd}/PROJECTION_DISPATCH",:int) unless head && pd !~ /(?:\A|\s)#{f}=/}}}}
  ta=i[:boxes].values.map{|x|x[:times]};need((ta[0][0]-ta[1][0]).abs<=120 && (ta[0][1]-ta[1][1]).abs<=120,"ranks' start/end times differ by more than 120 s in #{dir}")
  i[:start]=ta.map(&:first).min;i[:end]=ta.map(&:last).max
  t=readf("#{dir}/TEMPS.txt");i[:temps]={}
  rows=t.lines.map(&:strip).reject(&:empty?);need(rows.all?{|l|l=~/\A(#{pair.map{|b|Regexp.escape(b)}.join("|")}) GPU_TEMP (before|after) (\d{1,3})\z/ && Integer($3,10).between?(0,130)},"TEMPS.txt in #{dir} has a line outside '<box> GPU_TEMP before|after <0..130>'");need(rows.size==4,"TEMPS.txt in #{dir} has #{rows.size} lines, need 4")
  pair.each{|b| %w[before after].each{|w| v=t.scan(/^#{b} GPU_TEMP #{w} (\d+)\s*$/).flatten;need(v.size==1,"TEMPS.txt in #{dir}: #{v.size} '#{b} #{w}' lines");i[:temps]["#{b} #{w}"]=Integer(v[0],10)}}
  b0=i[:boxes][pair[0]][:dir];i[:sha]={}
  JOBS.each{|j,(_,n)| bytefiles(n).each{|f| ref=sha("#{b0}/#{j}/run1/#{f}");i[:sha]["#{j}/#{f}"]=ref
    pair.each{|b| (1..3).each{|k| p="#{i[:boxes][b][:dir]}/#{j}/run#{k}/#{f}";need(sha(p)==ref,"bytes differ inside the load: #{p} vs #{b0}/#{j}/run1/#{f}")}}}}
  r0=b0
  i[:decode]=JOBS.keys.to_h{|j| [j,DECODE_RUNS[j].map{|k| num(readf("#{r0}/#{j}/run#{k}/TIMING"),"decode_tokens_per_second","#{r0}/#{j}/run#{k}/TIMING")}]}
  i[:prompt]=%w[out1040 outlong].to_h{|j| [j,(1..3).map{|k| num(readf("#{r0}/#{j}/run#{k}/PREFILL_TIMING"),"prompt_seconds","#{r0}/#{j}/run#{k}/PREFILL_TIMING")}]}
  # v4 captures (VERIFY spec): raw graph_captures of sessions 2 and 3, rank 0, every length; decode_steps must equal the expected work
  i[:cap_raw]=JOBS.keys.to_h{|j| [j,[2,3].map{|k| p="#{r0}/#{j}/run#{k}";st=num(readf("#{p}/TIMING"),"decode_steps","#{p}/TIMING",:int)
    need(st==STEPS_EXPECTED[j],"#{p}/TIMING decode_steps #{st}, expected #{STEPS_EXPECTED[j]}");num(readf("#{p}/COUNTS"),"graph_captures","#{p}/COUNTS",:int).to_i}]}
  i[:captures]=%w[glm-tokens64 out127].to_h{|j| [j,(1..3).map{|k| p="#{r0}/#{j}/run#{k}";num(readf("#{p}/COUNTS"),"graph_captures","#{p}/COUNTS",:int).fdiv(num(readf("#{p}/TIMING"),"decode_steps","#{p}/TIMING",:int))}]}
  i[:bf16]=%w[project_bf16_ordered project_f32_ordered].to_h{|key| v=(1..3).map{|k| p="#{r0}/glm-tokens64/run#{k}/PROJECTION_DISPATCH";t=readf(p)
      (head && t !~ /(?:\A|\s)#{key}=/) ? nil : num(t,key,p,:int).fdiv(num(t,"decode_steps",p,:int))}
    need(v.all? || v.none?,"#{key} present in some runs only in #{dir}");[key,v.all? ? med(v) : nil]}
  i[:ready]=pair.map{|b| lg=i[:boxes][b][:log];m=lg.scan(/^RESIDENT READY rank=\d+ .*?\bbf16_copy_bytes=(\d+) bf16_copy_matrices=(\d+)\b/)
    need(m.size==1 || (head && m.empty? && lg.scan(/^RESIDENT READY /).size==1),"RESIDENT READY bf16 fields: #{m.size} lines on #{b} in #{dir}");m.empty? ? nil : m[0].map{|x|Integer(x,10)}}
  need(i[:ready].uniq.size==1,"RESIDENT READY differs between ranks in #{dir}")
  i[:counts]=JOBS.keys.to_h{|j| p="#{r0}/#{j}/run1/COUNTS";t=readf(p);[j,%w[graph_replays graph_captures stream_waits host_launches].to_h{|k|[k,num(t,k,p,:int).to_i]}]}
  sess=[];cur=nil;i[:boxes][pair[0]][:log].each_line{|l| m=l.match(/^RESIDENT TOKEN rank=0 repeat=\d+ at=(\d+) id=\d+ elapsed_seconds=([\d.]+)(\s|$)/) or next;(cur=[];sess<<cur) if m[1]=="0";need(cur,"TOKEN line before a session start in #{dir}");cur<<Float(m[2])}
  i[:steps]={}
  {"out1040"=>31,"outlong"=>47}.each{|j,n| ss=sess.map{|s|s.each_cons(2).map{|a,b|(b-a)*1000}}.select{|d|d.size==n};need(ss.size>=3,"#{j}: #{ss.size} step sessions of #{n} steps (need 3) in #{dir}")
    ss=ss[1..2];i[:steps][j]=[med(ss.flatten),ss.sum{|d| m=med(d);d[3..].count{|x|x>=m*1.25}}]}
  i[:boundary]={};i[:repeats]={};calls=[];tok=[]
  i[:boxes][pair[0]][:log].each_line{|l|
    if l=~/^RESIDENT DETAIL name=residual_begin calls=(\d+)\s/ then calls<<Integer($1,10)
    elsif l.start_with?("RESIDENT TOKEN rank=0 ") then tok<<l
    elsif l=~/RESIDENT JOB PASS .*out=(\S+)/ then j=$1
      if JOBS[j] then need(calls.size==3,"#{j}: #{calls.size} residual_begin lines (need 3) in #{dir}");need(!i[:boundary][j],"#{j} section twice in #{dir}")
        i[:boundary][j]=calls.each_with_index.map{|c,k| c.fdiv(num(readf("#{r0}/#{j}/run#{k+1}/TIMING"),"decode_steps","#{r0}/#{j}/run#{k+1}/TIMING",:int))};i[:repeats][j]=tok end
      calls=[];tok=[] end}
  JOBS.each_key{|j|need(i[:boundary][j],"no residual_begin section for #{j} in #{dir}")}
  i
end
def golden_check(head_sha)
  n=0
  %w[glm-tokens64 out127 out1040].each{|j| (1..3).each{|k| g="#{GOLDEN}/#{j}/run#{k}";need(File.directory?(g),"golden #{g} missing")
    bytefiles(JOBS[j][1]).each{|f| need(sha("#{g}/#{f}")==head_sha["#{j}/#{f}"],"HEAD differs from golden #{g}/#{f}");n+=1}
    need(!File.exist?("#{g}/#{JOBS[j][1]}-logits.f32"),"golden #{g} holds extra logits")}}
  lsum=readf("#{LONGREF}/logits.sha256")
  (0..16).each{|k| ref=lsum.scan(/^(\h{64})  \.\/base-long17\/run1\/#{k}-logits\.f32$/).flatten;need(ref.size==1,"long reference: #{ref.size} hashes for logits #{k}")
    need(ref[0]==head_sha["outlong/#{k}-logits.f32"],"HEAD differs from long reference logits #{k}");n+=1}
  ids=readf("#{LONGREF}/base-long17/run1/OUTPUT_IDS").lines;need(ids.size>=17,"long reference OUTPUT_IDS has #{ids.size} lines")
  need(sha("#{LONGREF}/base-long17/run1/PROMPT_LOGITS")==head_sha["outlong/PROMPT_LOGITS"],"HEAD long prompt logits differ from the long reference")
  [n,ids.first(17)]
end
def host_state(log,rev)
  s=rev[0,9];return [:fail,"host log #{log} missing"] unless File.file?(log)
  t=File.read(log,encoding:"binary")
  bad=t.lines.grep(/^(TRAIN HOST FAILED|GATE FAIL|MAC CHECKS FAILED|STOPPED)/);return [:fail,"host log #{File.basename(log)} has '#{bad[0].strip[0,80]}'"] if bad.any?
  lines=t.lines.map(&:strip);last=lines.rindex{|l|l.start_with?("TRAIN HOST")};return [:pending,"host gate running (#{File.basename(log)})"] unless last
  return [:fail,"host: last TRAIN HOST line is '#{lines[last][0,60]}' (need TRAIN HOST PASS #{s})"] unless lines[last]=="TRAIN HOST PASS #{s}"
  pre=lines[0...last];return [:fail,"host: no 'PAYLOAD READY #{s}' and 'MAC BUILD AND CHECKS PASS #{s}' before the verdict"] unless pre.include?("PAYLOAD READY #{s}") && pre.include?("MAC BUILD AND CHECKS PASS #{s}")
  [:pass,"TRAIN HOST PASS #{s} (#{File.basename(log)})"]
end
def ancestor(a,b) system("git","-C",REPO,"merge-base","--is-ancestor",a,b,out: File::NULL,err: File::NULL) end
def pct(a,b) (b/a-1)*100 end
hh=nil
begin
  hh=heads.map{|d| load_info(d,headrev,pair,true)}
  hh[1..].each{|h| h[:sha].each{|k,v| need(v==hh[0][:sha][k],"head loads differ in bytes: #{k}")}}
  gold,longids=golden_check(hh[0][:sha])
  need(readf("#{hh[0][:boxes][pair[0]][:dir]}/outlong/run1/OUTPUT_IDS").lines.first(17)==longids,"HEAD long ids differ from the long reference")
  puts "GATECHECK head #{headrev[0,9]} #{hh.size} load(s) #{hh.map{|h|File.basename(File.dirname(h[:dir]))}.join(",")}: inventory, identity, bytes inside each load, golden (#{gold} files; 4,158 reference covers 17 of 48 logits) PASS; times #{hh.map{|h|Time.at(h[:start]).strftime("%H:%M:%S")+"-"+Time.at(h[:end]).strftime("%H:%M:%S")}.join(", ")}; temps #{hh.map{|h|h[:temps].map{|k,v|"#{k} #{v}"}.join(" ")}.join("; ")}"
rescue StandardError=>e
  puts "GATECHECK HEAD INVALID: #{e.message}. Every rung FAILS (no valid reference).";rungs.each{|r|puts "RUNG #{r[:name]} #{r[:rev][0,9]} (#{r[:label]}) VERDICT FAIL: head invalid"};puts "HIGHEST PASSING RUNG: none. FIRST RUNG NOT PASSING: #{rungs[0][:name]}.";finish(1)
end
prev=headrev;blocked=false;best=nil;firstbad=nil
rungs.each{|r| fails=[];voids=[];notes=[];pending_r=false
  begin
    need(ancestor(headrev,r[:rev]),"#{r[:rev][0,9]} does not descend from the head #{headrev[0,9]}")
    need(ancestor(prev,r[:rev]),"#{r[:rev][0,9]} does not descend from the rung below (#{prev[0,9]})")
    tt=r[:loads].map{|d| load_info(d,r[:rev],pair)}
    tt.each{|t| hh[0][:sha].each{|k,v| need(t[:sha][k]==v,"bytes differ from the head: #{t[:dir]} #{k}")}}
    notes<<"BYTES PASS (#{tt.size} load(s) x #{hh[0][:sha].size} files vs head, both ranks, runs 1-3 equal inside every load)"
    d=JOBS.map{|j,(name,_)| hv=hh.flat_map{|h|h[:decode][j]};tv=tt.flat_map{|t|t[:decode][j]};hm=med(hv);tm=med(tv);c=pct(hm,tm);hs=(hv.max-hv.min)/hm*100
      fails<<"decode #{name} #{"%+.2f"%c}% below floor -#{FLOOR[j]}%" if c < -FLOOR[j];fails<<"RERUN #{name}: head session spread #{"%.2f"%hs}% over 4%" if hs>4
      "#{name} #{"%.3f"%hm} -> #{"%.3f"%tm} #{"%+.2f"%c}% (floor #{FLOOR[j]}, head spread #{"%.2f"%hs}%)"}
    notes<<"DECODE (rank 0; runs 1-3 at 15/127, 2-3 at 1,040/4,158): "+d.join("; ")
    p=%w[out1040 outlong].map{|j| hl=hh.map{|h|med(h[:prompt][j])};tl=tt.map{|t|med(t[:prompt][j])};hm=med(hl);tm=med(tl);c=pct(hm,tm)
      allow=hh.size==1 ? (j=="outlong" ? 1.5 : 1.0) : [[1.0,(hl.max-hl.min)/hm*100/2].max,2.0].min
      fails<<"prompt #{JOBS[j][0]} #{"%+.2f"%c}% over #{allow}%" if c>allow
      hsp=hh.map{|h|v=h[:prompt][j];(v.max-v.min)/med(v)*100}.max;fails<<"RERUN prompt #{JOBS[j][0]}: head session spread #{"%.2f"%hsp}% over 4%" if hsp>4
      "#{JOBS[j][0]} head #{hh.map{|h|h[:prompt][j].map{|x|"%.2f"%x}.join("/")}.join(" ")} | train #{tt.map{|t|t[:prompt][j].map{|x|"%.2f"%x}.join("/")}.join(" ")}: #{"%.3f"%hm} -> #{"%.3f"%tm} s #{"%+.2f"%c}% (allowance #{allow}%)"}
    notes<<"PROMPT (rank 0, sessions 1/2/3): "+p.join("; ")
    s=%w[out1040 outlong].map{|j| hm=med(hh.map{|h|h[:steps][j][0]});tm=med(tt.map{|t|t[:steps][j][0]});hsp=hh.map{|h|h[:steps][j][1]}.max;tsp=tt.map{|t|t[:steps][j][1]}.max
      fails<<"spikes #{JOBS[j][0]}: train #{tsp} over head #{hsp} + #{spike_margin}" if tsp>hsp+spike_margin
      "#{JOBS[j][0]} #{"%.1f"%hm} -> #{"%.1f"%tm} ms #{"%+.2f"%pct(hm,tm)}%, spikes head #{hh.map{|h|h[:steps][j][1]}.join("/")} train #{tt.map{|t|t[:steps][j][1]}.join("/")}"}
    notes<<"STEPS (sessions 2-3): "+s.join("; ")
    c=JOBS.map{|j,(name,_)| jd=CapturesRule.judge(hh.map{|h|h[:cap_raw][j]},tt.map{|t|t[:cap_raw][j]})
      fails<<"captures #{name} (raw sessions 2-3: "+jd[:results].each_with_index.reject{|x,_|x[:session_ok] && x[:total_ok]}.map{|x,i|"load #{i+1} #{x[:t].join("/")} total #{x[:total]} vs bound #{x[:bound]}, session limit #{jd[:limit]}"}.join("; ")+")" unless jd[:pass]
      CapturesRule.describe(name,jd)}
    notes<<"CAPTURES v4 (raw, sessions 2-3, expected steps 63/63/31/47): "+c.join(". ").gsub(". ","; ")
    notes<<"captures per step (info only, fractional): "+%w[glm-tokens64 out127].map{|j| "#{JOBS[j][0]} #{"%.2f"%med(hh.map{|h|med(h[:captures][j])})} -> #{"%.2f"%med(tt.map{|t|med(t[:captures][j])})}"}.join("; ")
    want=r[:opts]["bf16"] ? r[:opts]["bf16"].split(":").map{|x|Integer(x,10)} : hh[0][:ready][0]
    need(want && want.size==2,"no BF16 expectation: the head prints no RESIDENT READY bf16 fields and the rung names no bf16=<bytes>:<matrices>")
    need(r[:opts]["bf16"] || %w[project_bf16_ordered project_f32_ordered].all?{|k|hh[0][:bf16][k]},"no per-step BF16 expectation from the head")
    tt.each{|t| fails<<"READY bf16 #{t[:ready][0].join("/")} != expected #{want.join("/")}" if t[:ready][0]!=want
      %w[project_bf16_ordered project_f32_ordered].each{|k| fails<<"#{k}/step #{t[:bf16][k]} != head #{hh[0][:bf16][k]}" if !r[:opts]["bf16"] && (t[:bf16][k]-hh[0][:bf16][k]).abs>1e-9}}
    hb=hh[0][:bf16];notes<<"BF16: READY #{tt.map{|t|t[:ready][0].join("/")}.join(" ")} (expected #{want.join("/")}, both ranks); per step at 15: bf16 #{hb["project_bf16_ordered"] ? "%.1f"%hb["project_bf16_ordered"] : "n/a"} -> #{tt.map{|t|"%.1f"%t[:bf16]["project_bf16_ordered"]}.join("/")}, f32 #{hb["project_f32_ordered"] ? "%.1f"%hb["project_f32_ordered"] : "n/a"} -> #{tt.map{|t|"%.1f"%t[:bf16]["project_f32_ordered"]}.join("/")}"
    k=JOBS.map{|j,(name,_)| h=hh[0][:counts][j];tt.each{|t| fails<<"host_launches #{name} #{h["host_launches"]} -> #{t[:counts][j]["host_launches"]}" if t[:counts][j]["host_launches"]!=h["host_launches"]}
      "#{name}: "+h.keys.map{|x|"#{x} #{h[x]}->#{tt.map{|t|t[:counts][j][x]}.join("/")}"}.join(" ")}
    notes<<"COUNTERS (run 1, head vs each train load; replays reported, not checked): "+k.join("; ")
    b=JOBS.map{|j,(name,_)| hv=hh[0][:boundary][j];tt.each{|t| fails<<"boundary calls #{name} #{t[:boundary][j].inspect} != head #{hv.inspect}" if t[:boundary][j].size!=hv.size || t[:boundary][j].zip(hv).any?{|x,y|(x-y).abs>1e-9}};"#{name} #{hv.map{|x|"%.1f"%x}.uniq.join("/")}->#{tt.map{|t|t[:boundary][j].map{|x|"%.1f"%x}.uniq.join("/")}.join(" ")}"}
    notes<<"BOUNDARY CALLS/step (residual_begin): "+b.join("; ")
    if r[:opts]["counter"]=="1"
      rp=tt.each_with_index.flat_map{|t,ti| JOBS.map{|j,(name,_)| tl=t[:repeats][j];need(tl && tl.size>0,"no RESIDENT TOKEN lines for #{j}");acc=Hash.new(0)
        tl.each{|l| %w[table_replays table_repeats table_repeat_bytes table_bytes].each{|f| v=l.scan(/ #{f}=(\d+)(?=\s|$)/).flatten;need(v.size==1,"counter field #{f} #{v.size} times on a RESIDENT TOKEN line (#{j})");acc[f]+=Integer(v[0],10)}
          l.scan(/ (table_\w*stable\w*)=(\d+)(?=\s|$)/).each{|f,v|acc[f]+=Integer(v,10)}}
        "load #{ti+1} #{name} repeats #{acc["table_repeats"]}/#{acc["table_replays"]} = #{acc["table_replays"]>0 ? "%.4f"%(acc["table_repeats"].to_f/acc["table_replays"]) : "n/a"}, bytes #{acc["table_repeat_bytes"]}/#{acc["table_bytes"]}"+(acc.keys.grep(/stable/).any? ? " ("+acc.keys.grep(/stable/).map{|f|"#{f} #{acc[f]}"}.join(" ")+")" : "")}}
      notes<<"TABLE REPEATS (rank 0, all sessions): "+rp.join("; ")
    end
    tt.each{|t| %w[before after].each{|w| m=pair.map{|bx|(t[:temps]["#{bx} #{w}"]-hh[0][:temps]["#{bx} #{w}"]).abs}.max;voids<<"temperature #{w} differs #{m} C from the head (#{File.basename(File.dirname(t[:dir]))})" if m>=5}
      age=t[:start]-hh[-1][:end];voids<<"age #{age.round} s (train start minus head end) outside 0..3600 (#{File.basename(File.dirname(t[:dir]))})" unless age>=0 && age<=3600}
    notes<<"TEMPS train #{tt.map{|t|t[:temps].map{|kk,v|"#{kk} #{v}"}.join(" ")}.join("; ")}; TIMES train #{tt.map{|t|Time.at(t[:start]).strftime("%H:%M:%S")+"-"+Time.at(t[:end]).strftime("%H:%M:%S")}.join(", ")}, start minus head end #{tt.map{|t|(t[:start]-hh[-1][:end]).round}.join("/")} s"
    hs,ht=host_state(r[:host],r[:rev]);notes<<"HOST: #{ht}";fails<<ht if hs==:fail;pending_r=(hs==:pending)
  rescue Bad=>e then fails<<"INVALID EVIDENCE: #{e.message}"
  rescue StandardError=>e then fails<<"TOOL ERROR (fail closed): #{e.class}: #{e.message[0,200]}"
  end
  own=fails.any? ? "FAIL: "+fails.join("; ") : voids.any? ? "VOID: "+voids.join("; ") : pending_r ? "PENDING: host gate not finished" : "PASS"
  v=blocked ? "BLOCKED (a lower rung did not pass); own checks: #{own}" : own
  puts "RUNG #{r[:name]} #{r[:rev][0,9]} (#{r[:label]}) loads #{r[:loads].map{|x|File.basename(File.dirname(x))+"/"+File.basename(x)}.join(",")}. "+notes.join(". ")+". VERDICT #{v}"
  if v=="PASS" then best=r else blocked=true;firstbad||=r.merge(v:v) end
  prev=r[:rev]}
puts "HIGHEST PASSING RUNG: #{best ? "#{best[:name]} #{best[:rev][0,9]} (#{best[:label]})" : "none"}. FIRST RUNG NOT PASSING: #{firstbad ? "#{firstbad[:name]} #{firstbad[:rev][0,9]}, ticket #{firstbad[:label]}" : "none"}."
finish(firstbad.nil? ? 0 : (firstbad[:v].start_with?("PENDING") ? 2 : 1))
