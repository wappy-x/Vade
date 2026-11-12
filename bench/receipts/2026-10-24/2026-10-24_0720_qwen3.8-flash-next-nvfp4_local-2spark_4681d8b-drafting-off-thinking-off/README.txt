Engine 4681d8b2c83f3df09926046dd6bd8b7fd124ce69 (measured build); exact two-Spark layer pipeline on spark-a + spark-b.
Drafting OFF, thinking OFF. Status PASS.
Receipts: <path>
Cells (this card): cold 8K actual-mode TTFT median n=6/6, one-chat prose median n=3/3 (1024 cap), C8 median 8 rounds, C16 median 8 rounds (256 cap, +1 warmup each).
C16: 16 offered chats; pipeline admits active=10 this start, excess HTTP 503 retried with identical sealed body; tok/s over first-attempt to last-end wall. Attempts receipted per stream.
Token IDs ON vs OFF (measured pairs): 201/201 identical; smoke_equal=True. Groups: {"c16": {"identical": 128, "total": 128}, "c16_warmup": {"identical": 16, "total": 16}, "c8": {"identical": 64, "total": 64}, "c8_warmup": {"identical": 8, "total": 8}, "cold": {"identical": 6, "total": 6}, "prose": {"identical": 3, "total": 3}, "smoke": {"identical": 1, "total": 1}}.
Median SM clock (union of measured windows): {"spark-a": 2457.0, "spark-b": 2463.0} MHz (spark-a/spark-b).
Every greedy request uses seed 20260905 (concurrency adds stream index) and cache=false; cold prompts are exact 8192 tokens with unique nonces. Greedy IDs are seed independent.
