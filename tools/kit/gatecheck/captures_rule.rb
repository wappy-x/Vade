# Lane M captures rule for gatecheck v4 (2026-11-06 22:1x), exactly as lane VERIFY's spec
# notes/reviews/verify-captures-rule-2026-11-06.md. Pure arithmetic on raw integer graph_captures of sessions 2 and 3; no floats,
# no rounding, margin zero. Used by tools/gatecheck.rb; unit-tested by tools/captures-rule-test.rb.
#   heads:  one [s2,s3] pair per declared head load (1 or 2 loads)
#   trains: one [s2,s3] pair per candidate load (each judged separately, never pooled)
# H = all head counts; spread = max(H) - min(H); session_limit = max(H) + spread;
# each candidate: both sessions <= session_limit AND n * (t2+t3) <= sum(H) + n * spread  (n = number of head loads).
module CapturesRule
  def self.count?(x) x.is_a?(Integer) && x>=0 end
  def self.judge(heads,trains)
    raise ArgumentError,"captures rule: 1 or 2 head loads, got #{heads.size}" unless [1,2].include?(heads.size)
    raise ArgumentError,"captures rule: no candidate load" if trains.empty?
    (heads+trains).each{|p| raise ArgumentError,"captures rule: each load needs exactly sessions 2 and 3 as non-negative integers, got #{p.inspect}" unless p.is_a?(Array) && p.size==2 && p.all?{|x|count?(x)}}
    h=heads.flatten;n=heads.size;spread=h.max-h.min;limit=h.max+spread;sum=h.sum
    res=trains.map{|t| tot=t.sum;bad=t.each_with_index.select{|x,_|x>limit}.map{|_,i|i+2}
      {t:t,total:tot,failed_sessions:bad,session_ok:bad.empty?,total_ok:n*tot<=sum+n*spread,bound:(n==1 ? "#{sum+spread}" : "(#{sum}+2*#{spread})/2=#{Rational(sum+2*spread,2)}")}}
    {heads:heads,n:n,min:h.min,max:h.max,spread:spread,limit:limit,head_total_per_load:Rational(sum,n),results:res,pass:res.all?{|r|r[:session_ok] && r[:total_ok]}}
  end
  def self.describe(name,j)
    "#{name}: head #{j[:heads].map{|p|p.join("/")}.join(" ")} (session 1 excluded), min #{j[:min]} max #{j[:max]} spread #{j[:spread]}, session limit #{j[:limit]}, head total per load #{j[:head_total_per_load].denominator==1 ? j[:head_total_per_load].numerator : j[:head_total_per_load]}; "+
      j[:results].each_with_index.map{|r,i| "load #{i+1} #{r[:t].join("/")} total #{r[:total]} vs bound #{r[:bound]}: session check #{r[:session_ok] ? "pass" : "FAIL (session #{r[:failed_sessions].join(",")})"}, total check #{r[:total_ok] ? "pass" : "FAIL"}"}.join("; ")+" (raw captures, margin 0)"
  end
end
