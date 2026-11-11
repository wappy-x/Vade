# Link checks

> Historical component snapshot. The counts and tools below describe the
> original check, before the monorepo import. They do not establish current
> link status. Known failed URLs remain as historical text below. On October 13,
> 2026, both recorded Oliaro portfolio addresses failed DNS resolution. Use the
> [Credits](../../docs/CREDITS.md) profile link instead.

**All 355 outside links in the public pages answered: 349 with HTTP 200, and
6 refused automated requests (HTTP 403 or 405).** Those six were checked by
other means, and every profile link on [Credits](../../docs/CREDITS.md) was checked
against the profile itself.

This page records both checks and how to repeat them, for maintainers. The
public pages are the README, `NOTICE`, `docs/`, and the crate, tool and
fixture guides: the Markdown pages listed in `scripts/export-allowlist.txt`.

<details>
<summary><b>Terms on this page</b>: HTTP status, redirect, HEAD and GET request, DOI</summary>

<dl>
<dt>HTTP status</dt>
<dd>The number a web server answers with.</dd>
<dt>200</dt>
<dd>OK</dd>
<dt>403</dt>
<dd>The server refused the request.</dd>
<dt>405</dt>
<dd>The server refuses that kind of request.</dd>
<dt>Redirect</dt>
<dd>The server sends the request on to another address.</dd>
<dt>HEAD request</dt>
<dd>Asks a server for a page's status without downloading the page.</dd>
<dt>GET request</dt>
<dd>Asks for the page itself; used only where HEAD was refused.</dd>
<dt>DOI</dt>
<dd>A permanent link for a published paper; it redirects to the publisher.</dd>
</dl>

The check follows every redirect.

</details>

## In short

**A reachable link is not a verified fact.** HTTP 200 means the page
answered; it does not prove the page's contents, a paper's authors or an
API's behavior.

Checked on 2026-10-04, from a development computer, not a Spark.

| Result | Unique links |
| --- | ---: |
| Reachable (HTTP 200) | 349 |
| Refused (HTTP 403 or 405) | 6 |
| No answer | 0 |
| **Total** | **355** |

- Reachable counts the final answer, after redirects.
- The six refusals are under
  [Links checked by other means](#links-checked-by-other-means).
- Paper authors were checked separately: all 36 arXiv papers cited in
  [Credits](../../docs/CREDITS.md) had their titles and author lists compared with
  arXiv's own records, through `export.arxiv.org`, in the previous review.

## Profile links on Credits

**Every person and team link on [Credits](../../docs/CREDITS.md) was checked against
the profile itself, on 2026-10-04.**

- **GitHub.** The profile page at `https://github.com/<login>` was fetched
  and its displayed name, listed website and listed X account were read. A
  login is linked only when the displayed name matches the credited person,
  or when the login itself is the credited name. An X handle or website is
  listed only when that profile lists it.
- **Hugging Face.** `https://huggingface.co/api/users/<name>/overview` (or
  `api/organizations/<name>/overview`) was fetched and its full name
  compared with the credited name. Accounts with no full name are not
  linked.
- **Nothing guessed.** People with no verified profile are credited by name and
  paper only.

**People**: 62 verified GitHub profiles, and what each one confirmed

Each entry gives the GitHub login and what its profile confirmed, then any
Hugging Face account and what its profile showed. A person with no Hugging
Face entry has no Hugging Face link.

<dl>
<dt>Tri Dao</dt>
<dd>GitHub <code>tridao</code>: name, website tridao.me. Hugging Face: tridao
(full name matches).</dd>
<dt>Daniel Y. Fu</dt>
<dd>GitHub <code>DanFu09</code>: name (Dan Fu), website danfu.org, X
<code>@realDanFu</code>. Hugging Face: danfu09 (Dan Fu).</dd>
<dt>Vijay Thakkar</dt>
<dd>GitHub <code>thakkarV</code>: name.</dd>
<dt>Maxim Milakov</dt>
<dd>GitHub <code>milakov</code>: name, website milakov.org.</dd>
<dt>Natalia Gimelshein</dt>
<dd>GitHub <code>ngimel</code>: name.</dd>
<dt>Francisco Massa</dt>
<dd>GitHub <code>fmassa</code>: name.</dd>
<dt>Grigory Sizov</dt>
<dd>GitHub <code>sgrigory</code>: name; the listed site is a LinkedIn page, not
linked.</dd>
<dt>Horace He</dt>
<dd>GitHub <code>Chillee</code>: name, website horace.io, X
<code>@cHHillee</code>. Hugging Face: Chillee (full name matches).</dd>
<dt>Gyeong-In Yu</dt>
<dd>GitHub <code>gyeongin</code>: name, website gyeongin.github.io.</dd>
<dt>Woosuk Kwon</dt>
<dd>GitHub <code>WoosukKwon</code>: name, website woosuk.me, X
<code>@woosuk_k</code>. Hugging Face: account exists with no full name; not
linked.</dd>
<dt>Zhuohan Li</dt>
<dd>GitHub <code>zhuohan123</code>: name, website zhuohan.li, X
<code>@zhuohan123</code>.</dd>
<dt>Siyuan Zhuang</dt>
<dd>GitHub <code>suquark</code>: name.</dd>
<dt>Ying Sheng</dt>
<dd>GitHub <code>Ying1123</code>: name.</dd>
<dt>Lianmin Zheng</dt>
<dd>GitHub <code>merrymercy</code>: name, website lmzheng.net, X
<code>@lm_zheng</code>. Hugging Face: lmzheng (full name "Lianmin", same handle
as the listed website).</dd>
<dt>Cody Hao Yu</dt>
<dd>GitHub <code>comaniac</code>: name (Cody Yu), website
comaniac.github.io.</dd>
<dt>Joseph E. Gonzalez</dt>
<dd>GitHub <code>jegonzal</code>: name, website at Berkeley, X
<code>@profjoeyg</code>.</dd>
<dt>Hao Zhang</dt>
<dd>GitHub <code>zhisbug</code>: name, website haozhang.ai, X
<code>@haozhangml</code>.</dd>
<dt>Ion Stoica</dt>
<dd>GitHub <code>istoica</code>: first name only, plus the Berkeley website in
his name.</dd>
<dt>Liangsheng Yin</dt>
<dd>GitHub <code>hnyls2002</code>: name, X <code>@lsyincs</code>; the listed
website lsyin.me did not answer.</dd>
<dt>Zhiqiang Xie</dt>
<dd>GitHub <code>xiezhq-hermann</code>: name, website zhiqiangxie.com, X
<code>@Zhiqiang_Xie</code>.</dd>
<dt>Zihao Ye</dt>
<dd>GitHub <code>yzh119</code>: name, website expye.com. Hugging Face: account
exists with no full name; not linked.</dd>
<dt>Lequn Chen</dt>
<dd>GitHub <code>abcdabcd987</code>: name, website abcdabcd987.com.</dd>
<dt>Ruihang Lai</dt>
<dd>GitHub <code>MasterJH5574</code>: name, website ruihanglai.com, X
<code>@ruihanglai</code>.</dd>
<dt>Apoorv Saxena</dt>
<dd>GitHub <code>apoorvumang</code>: name, website apoorvumang.github.io, X
<code>@apoorv_umang</code>. Hugging Face: apoorvumang (full name matches).</dd>
<dt>Xianzhen Luo</dt>
<dd>GitHub <code>Luowaterbi</code>: name, website luowaterbi.github.io.</dd>
<dt>Zhenyu He</dt>
<dd>GitHub <code>zhenyuhe00</code>: name, website zhenyuhe00.github.io.</dd>
<dt>Tianle Cai</dt>
<dd>GitHub <code>ctlllll</code>: name, website tianle.website, X
<code>@tianle_cai</code>.</dd>
<dt>Yichao Fu</dt>
<dd>GitHub <code>Viol2000</code>: name, website viol2000.github.io, X
<code>@fuyichao123</code>.</dd>
<dt>Peter Bailis</dt>
<dd>GitHub <code>pbailis</code>: name, website bailis.org, X
<code>@pbailis</code>.</dd>
<dt>Gabriele Oliaro</dt>
<dd>GitHub <code>goliaro</code>: name, website gabrieleoliaro.com.</dd>
<dt>KlyzhenkoVadim</dt>
<dd>GitHub <code>KlyzhenkoVadim</code>: the login is the credited name
(pull-request author).</dd>
<dt>T.J. Purtell</dt>
<dd>GitHub <code>tpurtell</code>: name.</dd>
<dt>Victor Cruz</dt>
<dd>GitHub <code>vcruz305</code>: name, X <code>@ViC305</code>. Hugging Face:
vcruz305 (full name matches).</dd>
<dt>Łukasz Kaiser</dt>
<dd>GitHub <code>lukaszkaiser</code>: name.</dd>
<dt>Jianlin Su</dt>
<dd>GitHub <code>bojone</code>: name, website kexue.fm (refuses automated
requests).</dd>
<dt>Bowen Peng</dt>
<dd>GitHub login bloc97 shows no name, so it is not linked. Hugging Face: bloc97
(full name matches).</dd>
<dt>Jeffrey Quesnelle</dt>
<dd>GitHub <code>jquesnelle</code>: name, website jeffq.com, X
<code>@theemozilla</code>. Hugging Face: account exists with no full name; not
linked.</dd>
<dt>Honglu Fan</dt>
<dd>GitHub <code>honglu2875</code>: name, website honglu.fan.</dd>
<dt>Enrico Shippole</dt>
<dd>GitHub <code>conceptofmind</code>: name, website teraflopai.com, X
<code>@EnricoShippole</code>. Hugging Face: conceptofmind (full name
matches).</dd>
<dt>Biao Zhang</dt>
<dd>GitHub <code>bzhangGo</code>: name.</dd>
<dt>Rico Sennrich</dt>
<dd>GitHub <code>rsennrich</code>: name; the listed university page answers HTTP
404, so it is not linked.</dd>
<dt>Songlin Yang</dt>
<dd>GitHub <code>sustcsonglin</code>: name, website sustcsonglin.github.io, X
<code>@SonglinYang4</code>. Hugging Face: account exists with no full name; not
linked.</dd>
<dt>Ali Hatamizadeh</dt>
<dd>GitHub <code>ahatamiz</code>: name, website ahatamiz.github.io, X
<code>@ahatamiz1</code>.</dd>
<dt>Bailin Wang</dt>
<dd>GitHub <code>berlino</code>: first name, website berlino.github.io, X
<code>@bailin_28</code>.</dd>
<dt>Yu Zhang</dt>
<dd>GitHub <code>yzhangcs</code>: name, website yzhang.site, X
<code>@yzhang_cs</code>.</dd>
<dt>Yikang Shen</dt>
<dd>GitHub <code>yikangshen</code>: name.</dd>
<dt>Yoon Kim</dt>
<dd>GitHub <code>yoonkim</code>: name, website at MIT CSAIL.</dd>
<dt>Barry Haddow</dt>
<dd>GitHub <code>bhaddow</code>: name.</dd>
<dt>Sebastiano Vigna</dt>
<dd>GitHub <code>vigna</code>: name, website vigna.di.unimi.it.</dd>
<dt>Mostofa Patwary</dt>
<dd>GitHub <code>mpatwary</code>: name.</dd>
<dt>Jared Casper</dt>
<dd>GitHub <code>jaredcasper</code>: name.</dd>
<dt>Yuhong Li</dt>
<dd>GitHub <code>leeyeehoo</code>: name, website leeyeehoo.github.io; the listed
X account answers HTTP 404, so it is not linked.</dd>
<dt>Hongwu Peng</dt>
<dd>GitHub <code>harveyp123</code>: name, website harveyp123.github.io.</dd>
<dt>Yuhui Li</dt>
<dd>GitHub <code>Liyuhui-12</code>: name (yuhuili).</dd>
<dt>Hongyang Zhang</dt>
<dd>GitHub <code>hongyanz</code>: name, website hongyanz.github.io, X
<code>@hongyangzh</code>.</dd>
<dt>Mostafa Elhoushi</dt>
<dd>GitHub <code>mostafaelhoushi</code>: name, X <code>@m_elhoushi</code>; the
listed site is an ORCID record, not linked.</dd>
<dt>Ash Hart</dt>
<dd>GitHub <code>ashhart</code>: first name, X <code>@ashxhart</code>; the
profile's Hugging Face link answers HTTP 404, so no HF link is listed.</dd>
<dt>JJ Eaton</dt>
<dd>GitHub <code>jayleaton</code>: name, website jayleaton.com, X
<code>@jayleaton</code>.</dd>
<dt>hughmadden</dt>
<dd>GitHub <code>hughmadden</code>: the login is the credited name (repository
owner).</dd>
<dt>Jürgen Schmied</dt>
<dd>GitHub <code>jschmied</code>: name.</dd>
<dt>Artem Matskevych</dt>
<dd>GitHub <code>Reederey87</code>: name, X <code>@Reederey</code>.</dd>
<dt>Christopher Owen</dt>
<dd>GitHub <code>christopherowen</code>: the login is the credited name
(repository owner); X <code>@christopherowen</code>.</dd>
<dt>Alfonso De Gregorio (secYOUre)</dt>
<dd>GitHub <code>secYOUre</code>: name, website pwnshow.com.</dd>
</dl>

**Teams and projects**: 30 GitHub organizations, and their listed sites

Each entry gives the GitHub organization and what its page lists, then any
Hugging Face account with its displayed name in brackets.

<dl>
<dt>Qwen team (Alibaba Cloud)</dt>
<dd>GitHub <code>QwenLM</code>: X <code>@Alibaba_Qwen</code>. Hugging Face: Qwen
(Qwen).</dd>
<dt>Google DeepMind</dt>
<dd>GitHub <code>google-deepmind</code>: website deepmind.com; linked as
deepmind.google. Hugging Face: google (Google).</dd>
<dt>OpenAI</dt>
<dd>GitHub <code>openai</code>: website openai.com (refuses automated requests).
Hugging Face: openai (OpenAI).</dd>
<dt>Mistral AI</dt>
<dd>GitHub <code>mistralai</code>: website mistral.ai. Hugging Face: mistralai
(Mistral AI).</dd>
<dt>Microsoft</dt>
<dd>GitHub <code>microsoft</code>: X <code>@OpenAtMicrosoft</code>. Hugging
Face: microsoft (Microsoft).</dd>
<dt>Hugging Face</dt>
<dd>GitHub <code>huggingface</code>: website huggingface.co, X
<code>@huggingface</code>. Hugging Face: huggingface (Hugging Face).</dd>
<dt>PyTorch</dt>
<dd>GitHub <code>pytorch</code>: website pytorch.org. Hugging Face: pytorch
(pytorch).</dd>
<dt>ByteDance Seed</dt>
<dd>GitHub <code>ByteDance-Seed</code>: website seed.bytedance.com. Hugging
Face: ByteDance-Seed (ByteDance Seed).</dd>
<dt>kernels-community</dt>
<dd>No GitHub organization. Hugging Face: kernels-community
(kernels-community).</dd>
<dt>Thinking Machines Lab</dt>
<dd>GitHub <code>thinking-machines-lab</code>: website thinkingmachines.ai.</dd>
<dt>vLLM</dt>
<dd>GitHub <code>vllm-project</code>: docs site docs.vllm.ai answers HTTP 200.
Hugging Face: vllm-project (vLLM).</dd>
<dt>SGLang</dt>
<dd>GitHub <code>sgl-project</code>: docs site docs.sglang.ai answers HTTP
200.</dd>
<dt>FlashInfer</dt>
<dd>GitHub <code>flashinfer-ai</code>: website flashinfer.ai.</dd>
<dt>DeepSeek</dt>
<dd>GitHub <code>deepseek-ai</code>: website deepseek.com. Hugging Face:
deepseek-ai (DeepSeek).</dd>
<dt>flash-linear-attention</dt>
<dd>GitHub <code>fla-org</code>: the org links hf.co/fla-hub. Hugging Face:
fla-hub (fla-hub).</dd>
<dt>Nous Research</dt>
<dd>GitHub <code>NousResearch</code>: website nousresearch.com, X
<code>@nousresearch</code>. Hugging Face: NousResearch (NousResearch).</dd>
<dt>FlexFlow</dt>
<dd>GitHub <code>flexflow</code>: website flexflow.ai.</dd>
<dt>Scaling Intelligence Lab</dt>
<dd>GitHub <code>ScalingIntelligence</code>: website
scalingintelligence.stanford.edu.</dd>
<dt>NVIDIA</dt>
<dd>GitHub <code>NVIDIA</code>: website nvidia.com. Hugging Face: nvidia
(NVIDIA).</dd>
<dt>linux-rdma</dt>
<dd>GitHub <code>linux-rdma</code>.</dd>
<dt>Rust</dt>
<dd>GitHub <code>rust-lang</code>: project site rust-lang.org answers HTTP
200.</dd>
<dt>LLVM</dt>
<dd>GitHub <code>llvm</code>: website llvm.org.</dd>
<dt>Mia's AI Lab</dt>
<dd>GitHub <code>MiaAI-Lab</code>: X <code>@MiaAI_lab</code>.</dd>
<dt>Local Inference Lab</dt>
<dd>GitHub <code>local-inference-lab</code>.</dd>
<dt>Kindling AI</dt>
<dd>GitHub <code>kindlingai</code>: website kindlingai.dev.</dd>
<dt>FujitsuPolycom</dt>
<dd>GitHub <code>FujitsuPolycom</code>.</dd>
<dt>turboderp (ExLlamaV3)</dt>
<dd>GitHub <code>turboderp-org</code>: X <code>@turboderp_</code>.</dd>
<dt>The Royal Lab (TabbyAPI)</dt>
<dd>GitHub <code>theroyallab</code>.</dd>
<dt>SafeAI Lab (EAGLE)</dt>
<dd>GitHub <code>SafeAILab</code>.</dd>
<dt>FasterDecoding (Medusa)</dt>
<dd>GitHub <code>FasterDecoding</code>.</dd>
<dt>Meta Research (LayerSkip)</dt>
<dd>GitHub <code>facebookresearch</code>.</dd>
</dl>

**Not linked**: logins tried and set aside

- **No name shown.** ermon, jayhshah, danthe3rd, bcatanzaro, nshazeer,
  FangyunWei: the GitHub page shows no name, so the login could not be tied
  to the person.
- **A different name.** mshoeybi, avaswani: the GitHub page shows a
  different name.
- **No such page.** jordan-juravsky, jordanjuravsky, xupeng-miao,
  ganeshbikshandi, zexuanzhong: no such GitHub page.
- **Hugging Face.** Accounts WoosukKwon, merrymercy, yzh119, jquesnelle,
  sustcsonglin, danthe3rd, Vontra: no full name, or the page does not exist,
  so none is linked.
- **NVIDIA forum.** Members Ank-Chy, andrewc_actual, mashie, ccatzin,
  ugurbas, robert287, adg1 and pontostroy: the forum's profile pages answer
  HTTP 403 to automated requests, so only their threads are linked.

## Links checked by other means

**Nine addresses could not be confirmed by a status code alone.** Six refuse
automated requests; three no longer worked and were removed from Credits.

**Confirmed another way**

- **405.** [https://milakov.org](https://milakov.org): HTTP 405 to HEAD;
  HTTP 200 to a GET request.
- **403.** [https://doi.org/10.1145/2660193.2660195](https://doi.org/10.1145/2660193.2660195):
  HTTP 403 from the publisher; confirmed through the Crossref record.
- **403.** [https://doi.org/10.1177/1094342005051521](https://doi.org/10.1177/1094342005051521):
  HTTP 403 from the publisher; confirmed through the Crossref record.

**Refuse automated requests**

- **403.** [https://kexue.fm](https://kexue.fm): HTTP 403 to HEAD and GET;
  the site refuses automated requests.
- **403.** [https://openai.com/](https://openai.com/): HTTP 403 to HEAD and
  GET; the site refuses automated requests.
- **403.** [https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):
  HTTP 403, also in a browser; not confirmed. The authors' paper on arXiv
  answers HTTP 200.

**Removed from Credits**

- **000.** `https://lsyin.me`: no answer (connection
  failed) to HEAD and GET.
- **404.** `https://www.cl.uzh.ch/sennrich`:
  HTTP 404 after a redirect.
- **404.** `https://x.com/yli3521`: HTTP 404; the
  account listed on the GitHub profile no longer exists.

The two DOI records were read through the Crossref API: "Fast splittable
pseudorandom number generators", Steele, Lea and Flood, OOPSLA 2014; and
"Optimization of Collective Communication Operations in MPICH", Thakur,
Rabenseifner and Gropp.

## Run the check yourself

**One redirect-following HEAD request per link, and a GET only for the few
sites that refuse HEAD.** No page bodies are kept.

### Do it yourself

1. From the checkout root, list every outside link in the public pages.
   Local addresses and placeholders are skipped.

   ```sh
   grep -E '\.md$|^NOTICE$' scripts/export-allowlist.txt \
     | grep -v '^docs/link-checks\.md$' \
     | xargs grep -hoE 'https?://[^][:space:])>"`<]+' \
     | sed 's/[.,;:]*$//; s/#.*//' \
     | grep -v -e localhost -e 'HOST:PORT' -e '\$' | sort -u > links.txt
   ```

2. Check each one, with a 25-second limit per link.

   ```sh
   while read -r URL; do
     printf '%s ' "$URL"
     curl -sIL --max-time 25 -o /dev/null -w '%{http_code} %{url_effective}\n' "$URL"
   done < links.txt > link-results.txt
   ```

3. List anything that did not answer 200.

   ```sh
   awk '$2 != 200' link-results.txt
   ```

4. To re-verify a GitHub profile, read its page and compare the displayed
   name with the credited person; to re-verify a Hugging Face profile, read
   the overview record:

   ```sh
   curl -s https://api.github.com/users/<login>
   curl -s https://huggingface.co/api/users/<name>/overview
   ```

   The GitHub API allows 60 unauthenticated requests per hour; the profile
   page `https://github.com/<login>` shows the same name, website and X
   account without that limit.

### Ask your coding agent

```text
In the Vade checkout, follow docs/link-checks.md, "Run the check yourself".
Use only the curl commands shown there; do not download page bodies or use
any other network tool. Report the totals and every link that did not
answer HTTP 200. Compare them with the results in docs/link-checks.md. Do
not edit files, commit or push.
```

## Complete results

**Every one of the 355 unique links, with its final status.** Each link was
requested once, and redirects were followed. Links with a `#` fragment share
the result of their page.

<details>
<summary><b>The 349 links that answered HTTP 200</b>, grouped by site</summary>

349 answered HTTP 200, grouped by site below; after each site comes the
path of every link on it. The other six are under
[Links checked by other means](#links-checked-by-other-means). The status is
the final answer, after any redirect; the next block lists the 27 redirects
and where each one ended.

<dl>
<dt>https://github.com (147)</dt>
<dd>/abcdabcd987, /ahatamiz, /apoorvumang, /apoorvumang/prompt-lookup-decoding,
/ashhart, /ashhart/TensorFold,
/ashhart/TensorFold/blob/17c73e189f5e6a5304cda7ea37f086f9c49b4788/src/tensorfold/families/qwen4_exp/cuda/decode.py,
/ashhart/TensorFold/tree/17c73e189f5e6a5304cda7ea37f086f9c49b4788, /berlino,
/bhaddow, /bojone, /ByteDance-Seed,
/bytedance-seed/veomni/blob/86c5ec1dbba8270206b1ab7f439285215ae4ec92/docs/key_features/mtp.md,
/bzhangGo, /Chillee, /christopherowen, /christopherowen/dgx-spark-memory-saver,
/comaniac, /conceptofmind, /ctlllll, /DanFu09, /deepseek-ai, /facebookresearch,
/FasterDecoding, /fla-org, /fla-org/flash-linear-attention, /flashinfer-ai,
/flashinfer-ai/flashinfer, /flexflow, /fmassa, /FujitsuPolycom,
/FujitsuPolycom/sparkring, /goliaro, /google-deepmind, /gyeongin, /harveyp123,
/hnyls2002, /honglu2875, /hongyanz, /huggingface, /huggingface/huggingface_hub,
/huggingface/tokenizers, /huggingface/transformers,
/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3/modeling_qwen3.py,
/huggingface/transformers/blob/9d8c3a24d434d4a958a23562f576126924358b7f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py,
/huggingface/transformers/blob/v5.18.0/src/transformers/integrations/finegrained_fp8.py,
/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py,
/huggingface/transformers/issues/48913, /hughmadden, /hughmadden/glm53f-afd,
/istoica, /jaredcasper, /jayleaton,
/jayleaton/glm53-tensorfold-spark/blob/main/docs/ROOFLINE.md, /jegonzal,
/jquesnelle, /jschmied,
/jschmied/qwen38-flash-next-gb10/blob/main/notes/what-generalises.md,
/kindlingai, /kindlingai/kindling-spark-os,
/kindlingai/kindling-spark-os/tree/5a8129d0837b6eb8aa469bb04d8e8fd7958e4d3e,
/KlyzhenkoVadim, /leeyeehoo, /linux-rdma, /linux-rdma/perftest,
/linux-rdma/rdma-core, /Liyuhui-12, /llvm,
/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsNVVM.td,
/local-inference-lab, /local-inference-lab/b12x,
/local-inference-lab/b12x/tree/a321f9a6e32f4119486c0ea9dd961595ffc87daf,
/lukaszkaiser, /Luowaterbi, /MasterJH5574, /merrymercy, /MiaAI-Lab,
/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/92bf731c3aac61927726ef0c422b21b42111f2c4/patches/0040-glm-parallel-deadlocks.patch,
/MiaAI-Lab/TensorFold, /microsoft, /milakov, /mistralai,
/mistralai/mistral-common, /mostafaelhoushi, /mpatwary, /ngimel, /NousResearch,
/NVIDIA, /NVIDIA/cutlass, /NVIDIA/nccl, /NVIDIA/nccl-tests,
/NVIDIA/TensorRT-LLM,
/NVIDIA/TensorRT-LLM/blob/a81da8a5a8380c4ec8c2d3eafabb8cfc55d7ad56/docs/source/features/kvcache.md,
/openai, /openai/gpt-2, /openai/gpt-oss, /openai/harmony,
/openai/openai-openapi, /openai/tiktoken, /pbailis, /pytorch, /pytorch/pytorch,
/QwenLM, /Reederey87,
/Reederey87/glm53-flash-exl3-2x-dgx-spark/blob/fd87969cb43a5ba66573bd2508bc05651fc1579c/docs/17-apc-tail-floor.md,
/rsennrich, /rust-lang, /rust-lang/rust, /SafeAILab, /ScalingIntelligence,
/secYOUre, /secYOUre/nvfp4bench, /sgl-project, /sgl-project/sglang, /sgrigory,
/suquark, /sustcsonglin, /thakkarV, /theroyallab, /theroyallab/tabbyAPI,
/thinking-machines-lab, /tpurtell, /tpurtell/ds41rt,
/tpurtell/ds41rt/blob/main/docs/dspark-bandwidth-policy.md, /tridao,
/turboderp-org, /turboderp-org/exllamav3, /vcruz305,
/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe, /vigna, /Viol2000,
/vllm-project, /vllm-project/vllm, /vllm-project/vllm-ascend/pull/7374,
/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py,
/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py,
/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py, /WoosukKwon,
/xiezhq-hermann, /yikangshen, /Ying1123, /yoonkim, /yzh119, /yzhangcs,
/zhenyuhe00, /zhisbug, /zhuohan123</dd>
<dt>https://huggingface.co (37)</dt>
<dd>/, /apoorvumang, /bloc97, /ByteDance-Seed, /Chillee, /conceptofmind,
/danfu09, /deepseek-ai, /docs/safetensors,
/docs/transformers/main/en/internal/rope_utils, /fla-hub, /google,
/google/gemma-4-31B-it, /huggingface, /kernels-community,
/kernels-community/finegrained-fp8/blob/3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/utils.py,
/lmzheng, /microsoft, /microsoft/phi-4, /mistralai, /NousResearch, /nvidia,
/openai, /pytorch, /Qwen,
/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca,
/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39,
/Qwen/Qwen3-32B/tree/9216db5781bf21249d130ec9da846c4624c16137, /Qwen/Qwen3-8B,
/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218,
/Qwen/Qwen3.5-0.8B, /Qwen/Qwen3.8-27B, /Qwen/Qwen3.8-27B-FP8,
/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13,
/tridao, /vcruz305, /vllm-project</dd>
<dt>https://arxiv.org (36)</dt>
<dd>/abs/1508.07909, /abs/1701.06538, /abs/1706.03762, /abs/1805.02867,
/abs/1909.08053, /abs/1910.07467, /abs/2002.05202, /abs/2104.09864,
/abs/2205.14135, /abs/2211.17192, /abs/2302.01318, /abs/2305.09781,
/abs/2305.13245, /abs/2307.08691, /abs/2309.00071, /abs/2309.06180,
/abs/2309.08168, /abs/2310.10537, /abs/2311.08252, /abs/2312.07104,
/abs/2401.10774, /abs/2401.15077, /abs/2402.02057, /abs/2402.05099,
/abs/2404.16710, /abs/2406.06484, /abs/2407.08608, /abs/2408.04678,
/abs/2408.08696, /abs/2409.12191, /abs/2409.19606, /abs/2412.06464,
/abs/2412.19437, /abs/2501.01005, /abs/2505.06708, /abs/2505.09388</dd>
<dt>https://x.com (33)</dt>
<dd>/ahatamiz1, /Alibaba_Qwen, /apoorv_umang, /ashxhart, /bailin_28, /cHHillee,
/christopherowen, /EnricoShippole, /fuyichao123, /haozhangml, /hongyangzh,
/huggingface, /jayleaton, /lm_zheng, /lsyincs, /m_elhoushi, /MiaAI_lab,
/nousresearch, /OpenAtMicrosoft, /pbailis, /profjoeyg, /realDanFu, /Reederey,
/ruihanglai, /SonglinYang4, /theemozilla, /tianle_cai, /turboderp_, /ViC305,
/woosuk_k, /yzhang_cs, /Zhiqiang_Xie, /zhuohan123</dd>
<dt>https://docs.nvidia.com (8)</dt>
<dd>/cuda/cuda-driver-api/, /cuda/cuda-driver-api/group__CUDA__GRAPH.html,
/cuda/cuda-driver-api/group__CUDA__VA.html,
/cuda/cuda-for-tegra-appnote/index.html, /cuda/parallel-thread-execution/,
/cutlass/latest/media/docs/cpp/blackwell_functionality.html,
/dgx/dgx-spark-porting-guide/porting/cuda.html,
/dgx/dgx-spark/spark-clustering.html</dd>
<dt>https://img.shields.io (5)</dt>
<dd>/badge/dependencies-no_outside_Rust_packages-e0e0e0?style=flat-square&amp;labelColor=1c1c1c,
/badge/license-Apache--2.0-e0e0e0?style=flat-square&amp;labelColor=1c1c1c,
/badge/NVIDIA-DGX_Spark_%2F_GB10-76b900?style=flat-square&amp;logo=nvidia&amp;logoColor=white&amp;labelColor=1c1c1c,
/badge/NVIDIA_math_libraries-none-e0e0e0?style=flat-square&amp;labelColor=1c1c1c,
/badge/Rust-from_scratch-b7410e?style=flat-square&amp;logo=rust&amp;logoColor=white&amp;labelColor=1c1c1c</dd>
<dt>https://www.unicode.org (5)</dt>
<dd>/license.txt, /Public/12.1.0/ucd/CompositionExclusions.txt,
/Public/12.1.0/ucd/UnicodeData.txt, /Public/16.0.0/ucd/UnicodeData.txt,
/ucd/</dd>
<dt>https://forums.developer.nvidia.com (4)</dt>
<dd>/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461,
/t/dgx-spark-direct-qsfp-connection-only-getting-13-16-gbps-instead-of-expected-200g-performance/370035,
/t/dgx-spark-gb10-faq/347344,
/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618</dd>
<dt>https://www.rfc-editor.org (4)</dt>
<dd>/rfc/rfc2104, /rfc/rfc4231, /rfc/rfc8259, /rfc/rfc9112</dd>
<dt>Other sites (70)</dt>
<dd>http://bailis.org, http://eecs.berkeley.edu/~jegonzal, http://horace.io/,
http://jeffq.com, http://leeyeehoo.github.io/, http://lmzheng.net,
http://vigna.di.unimi.it/, http://www.cs.berkeley.edu/~istoica/,
https://abcdabcd987.com, https://ahatamiz.github.io/,
https://apoorvumang.github.io, https://berlino.github.io/,
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf,
https://comaniac.github.io,
https://crfm.stanford.edu/2023/10/12/flashdecoding.html,
https://csrc.nist.gov/pubs/fips/180-4/upd1/final, https://deepmind.google,
https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/,
https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html,
https://docs.kernel.org/admin-guide/pm/cpuidle.html, https://docs.sglang.ai,
https://docs.vllm.ai, https://expye.com/about, https://flashinfer.ai/,
https://flashinfer.ai/2024/02/02/cascade-inference.html, https://flexflow.ai/,
https://gabrieleoliaro.com, https://gyeongin.github.io, https://haozhang.ai/,
https://harveyp123.github.io, https://honglu.fan, https://hongyanz.github.io/,
https://html.spec.whatwg.org/multipage/server-sent-events.html,
https://kindlingai.dev, https://llvm.org, https://llvm.org/docs/NVPTXUsage.html,
https://luowaterbi.github.io, https://mistral.ai, https://networkmanager.dev,
https://nousresearch.com, https://numpy.org, https://nvidia.com,
https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark,
https://people.csail.mit.edu/yoonkim/, https://prng.di.unimi.it/splitmix64.c,
https://pytorch.org, https://ruihanglai.com, https://rustup.rs,
https://scalingintelligence.stanford.edu/, https://seed.bytedance.com/,
https://sh.rustup.rs, https://sustcsonglin.github.io/,
https://thinkingmachines.ai,
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/,
https://tianle.website, https://tridao.me, https://viol2000.github.io/,
https://woosuk.me, https://www.danfu.org, https://www.deepseek.com/,
https://www.jayleaton.com/, https://www.pwnshow.com/, https://www.python.org/,
https://www.rust-lang.org, https://www.teraflopai.com/,
https://www.usenix.org/conference/osdi22/presentation/yu, https://yzhang.site,
https://zhenyuhe00.github.io, https://zhiqiangxie.com, https://zhuohan.li</dd>
</dl>

</details>

<details>
<summary><b>The 27 redirects</b>: where each one ended</summary>

<dl>
<dt><a href="http://bailis.org">http://bailis.org</a></dt>
<dd>Ended at <a href="https://www.bailis.org/">https://www.bailis.org/</a></dd>
<dt><a href="http://eecs.berkeley.edu/~jegonzal">http://eecs.berkeley.edu/~jegonzal</a></dt>
<dd>Ended at
<a href="https://people.eecs.berkeley.edu/~jegonzal/">https://people.eecs.berkeley.edu/~jegonzal/</a></dd>
<dt><a href="http://horace.io/">http://horace.io/</a></dt>
<dd>Ended at <a href="https://horace.io/">https://horace.io/</a></dd>
<dt><a href="http://leeyeehoo.github.io/">http://leeyeehoo.github.io/</a></dt>
<dd>Ended at
<a href="https://leeyeehoo.github.io/">https://leeyeehoo.github.io/</a></dd>
<dt><a href="http://vigna.di.unimi.it/">http://vigna.di.unimi.it/</a></dt>
<dd>Ended at
<a href="https://vigna.di.unimi.it/">https://vigna.di.unimi.it/</a></dd>
<dt><a href="http://www.cs.berkeley.edu/~istoica/">http://www.cs.berkeley.edu/~istoica/</a></dt>
<dd>Ended at
<a href="https://people.eecs.berkeley.edu/~istoica/">https://people.eecs.berkeley.edu/~istoica/</a></dd>
<dt><a href="https://abcdabcd987.com">https://abcdabcd987.com</a></dt>
<dd>Ended at
<a href="https://abcdabcd987.com/about/">https://abcdabcd987.com/about/</a></dd>
<dt><a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html">https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html</a></dt>
<dd>Ended at
<a href="https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html">https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html</a></dd>
<dt><a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html">https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html</a></dt>
<dd>Ended at
<a href="https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html">https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html</a></dd>
<dt><a href="https://docs.sglang.ai">https://docs.sglang.ai</a></dt>
<dd>Ended at <a href="https://docs.sglang.io/">https://docs.sglang.io/</a></dd>
<dt><a href="https://docs.vllm.ai">https://docs.vllm.ai</a></dt>
<dd>Ended at
<a href="https://docs.vllm.ai/en/latest/">https://docs.vllm.ai/en/latest/</a></dd>
<dt><a href="https://doi.org/10.1145/2660193.2660195">https://doi.org/10.1145/2660193.2660195</a></dt>
<dd>Ended at
<a href="https://dl.acm.org/doi/10.1145/2660193.2660195">https://dl.acm.org/doi/10.1145/2660193.2660195</a></dd>
<dt><a href="https://doi.org/10.1177/1094342005051521">https://doi.org/10.1177/1094342005051521</a></dt>
<dd>Ended at
<a href="https://journals.sagepub.com/doi/10.1177/1094342005051521">https://journals.sagepub.com/doi/10.1177/1094342005051521</a></dd>
<dt><code>https://gabrieleoliaro.com</code></dt>
<dd>Ended at
<code>https://www.gabrieleoliaro.com/</code></dd>
<dt><a href="https://honglu.fan">https://honglu.fan</a></dt>
<dd>Ended at
<a href="https://honglu2875.github.io/">https://honglu2875.github.io/</a></dd>
<dt><a href="https://huggingface.co/Chillee">https://huggingface.co/Chillee</a></dt>
<dd>Ended at
<a href="https://huggingface.co/chillee">https://huggingface.co/chillee</a></dd>
<dt><a href="https://huggingface.co/docs/safetensors">https://huggingface.co/docs/safetensors</a></dt>
<dd>Ended at
<a href="https://huggingface.co/docs/safetensors/index">https://huggingface.co/docs/safetensors/index</a></dd>
<dt><a href="https://kindlingai.dev">https://kindlingai.dev</a></dt>
<dd>Ended at
<a href="https://www.kindlingai.dev/">https://www.kindlingai.dev/</a></dd>
<dt><a href="https://nvidia.com">https://nvidia.com</a></dt>
<dd>Ended at
<a href="https://www.nvidia.com/en-us/">https://www.nvidia.com/en-us/</a></dd>
<dt><a href="https://seed.bytedance.com/">https://seed.bytedance.com/</a></dt>
<dd>Ended at
<a href="https://seed.bytedance.com/en/">https://seed.bytedance.com/en/</a></dd>
<dt><a href="https://tianle.website">https://tianle.website</a></dt>
<dd>Ended at
<a href="https://www.tianle.website/">https://www.tianle.website/</a></dd>
<dt><a href="https://www.danfu.org">https://www.danfu.org</a></dt>
<dd>Ended at <a href="https://danfu.org/">https://danfu.org/</a></dd>
<dt><a href="https://www.rfc-editor.org/rfc/rfc2104">https://www.rfc-editor.org/rfc/rfc2104</a></dt>
<dd>Ended at
<a href="https://www.rfc-editor.org/info/rfc2104/">https://www.rfc-editor.org/info/rfc2104/</a></dd>
<dt><a href="https://www.rfc-editor.org/rfc/rfc4231">https://www.rfc-editor.org/rfc/rfc4231</a></dt>
<dd>Ended at
<a href="https://www.rfc-editor.org/info/rfc4231/">https://www.rfc-editor.org/info/rfc4231/</a></dd>
<dt><a href="https://www.rfc-editor.org/rfc/rfc8259">https://www.rfc-editor.org/rfc/rfc8259</a></dt>
<dd>Ended at
<a href="https://www.rfc-editor.org/info/rfc8259/">https://www.rfc-editor.org/info/rfc8259/</a></dd>
<dt><a href="https://www.rfc-editor.org/rfc/rfc9112">https://www.rfc-editor.org/rfc/rfc9112</a></dt>
<dd>Ended at
<a href="https://www.rfc-editor.org/info/rfc9112/">https://www.rfc-editor.org/info/rfc9112/</a></dd>
<dt><a href="https://www.rust-lang.org">https://www.rust-lang.org</a></dt>
<dd>Ended at <a href="https://rust-lang.org/">https://rust-lang.org/</a></dd>
</dl>

</details>

## Local links

**All 530 relative Markdown links and 12 relative HTML `href` and `src`
links resolved on 2026-10-04.** The check ran after the last edit to the
pages above, with a small script that reads every Markdown page in the
export allowlist and counts every link occurrence outside code blocks.

- Every relative link points to a file that exists and is exported.
- Every link to a heading points to a heading that exists.
- Pages edited after that need checking again.
- No build, GPU benchmark or reference suite was rerun for this
  documentation-only work.

**Where next:** [Credits](../../docs/CREDITS.md) · [Export](export.md) ·
[Documentation index](README.md)
