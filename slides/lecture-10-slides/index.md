<section class="title">
  <div class="title-main">Lecture 10 – Information Retrieval and RAG</div>
  <div class="title-sub">CS40008.01: NLP & LLMs</div>
  <div class="title-meta">
    <div>Baojian Zhou</div>
    <div>School of Data Science</div>
    <div>Fudan University</div>
    <div>05/14/2026</div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="active">Motivation</li>
    <li class="muted">Information Retrieval (IR)</li>
    <li class="muted">Dense Retrieval</li>
    <li class="muted">Retrieval-Augmented Generation (RAG)</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">Motivation</div>
  <div class="ppt-line"></div>

  <div style="font-size:38px; line-height:1.45; font-weight:800; color:#1f4e9a; margin-bottom:20px;">
    An important function of LLMs is to fill <span style="color:#c00;">human information needs</span>
    by answering people's questions.
  </div>

  <div class="twocol" style="--left:52%;">
    <div>
      <ul class="outline-bullets med">
        <li>By 1961, computers answered baseball statistics questions</li>
        <li>IBM Watson won <em>Jeopardy!</em> in 2011 — at human level</li>
        <li>Modern search engines are deeply integrated with LLMs</li>
        <li>From <b>search engine</b> → <b>GPT-4 / GPT-5.1</b>: the boundary is blurring</li>
      </ul>
    </div>
    <div>
      <div style="border:2px solid #1f4e9a; border-radius:14px; padding:18px 20px; background:#f0f4fb; font-size:30px; line-height:1.5;">
        <div style="font-weight:900; color:#1f4e9a; margin-bottom:10px;">Timeline</div>
        <div style="margin-bottom:8px;">🔍 <b>1961</b> — First QA systems</div>
        <div style="margin-bottom:8px;">🏆 <b>2011</b> — Watson wins Jeopardy!</div>
        <div style="margin-bottom:8px;">🤖 <b>2020</b> — GPT-3 prompting</div>
        <div>🚀 <b>2025</b> — GPT-5.1, Claude 4, Gemini 2.5</div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Motivation — Factoid Questions</div>
  <div class="ppt-line"></div>

  <div style="font-size:34px; font-weight:700; margin-bottom:14px;">
    Simple <span style="color:#1f4e9a;">factoid questions</span> can be met with short, verifiable facts:
  </div>

  <div style="display:flex; flex-direction:column; gap:12px; margin-bottom:20px;">
    <div style="background:#f7f8fc; border-left:5px solid #1f4e9a; border-radius:8px; padding:12px 18px; font-size:32px; font-weight:700;">
      Where is Fudan University located?
    </div>
    <div style="background:#f7f8fc; border-left:5px solid #1f4e9a; border-radius:8px; padding:12px 18px; font-size:32px; font-weight:700;">
      Where does the energy in a nuclear explosion come from?
    </div>
    <div style="background:#f7f8fc; border-left:5px solid #1f4e9a; border-radius:8px; padding:12px 18px; font-size:32px; font-weight:700;">
      How to get a script ℓ in LaTeX?
    </div>
  </div>

  <div style="font-size:30px; line-height:1.5;">
    We can just <b>prompt an LLM</b> — it has processed a lot of facts during pretraining.
    But <span style="color:#c00; font-weight:900;">LLMs often give wrong answers to factual questions!</span>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Motivation — Hallucination</div>
  <div class="ppt-line"></div>

  <div style="font-size:36px; font-weight:800; color:#c00; margin-bottom:18px;">
    Hallucinations are common across all LLMs
  </div>

  <div style="font-size:30px; line-height:1.5; margin-bottom:18px;">
    When asked a direct, verifiable question (e.g., about a federal court case),
    LLMs confidently produce <b>plausible-sounding but incorrect</b> answers.
  </div>

  <div style="display:flex; gap:20px;">
    <div style="flex:1; border:2px solid #c00; border-radius:14px; padding:16px 18px; background:#fff8f8;">
      <div style="font-size:26px; font-weight:900; color:#c00; margin-bottom:8px;">❌ Hallucinated answer</div>
      <div style="font-size:26px; line-height:1.45; font-style:italic;">
        "The case was decided in 2018 by Judge Smith, who ruled in favor of the plaintiff based on §42 of the statute…"
      </div>
      <div style="font-size:22px; margin-top:10px; opacity:0.7;">(none of these details exist)</div>
    </div>
    <div style="flex:1; border:2px solid #1f4e9a; border-radius:14px; padding:16px 18px; background:#f0f4fb;">
      <div style="font-size:26px; font-weight:900; color:#1f4e9a; margin-bottom:8px;">Why does this happen?</div>
      <ul style="font-size:26px; line-height:1.55; padding-left:1em;">
        <li>LLMs predict <em>plausible</em> next tokens, not <em>true</em> facts</li>
        <li>Knowledge is compressed into weights — no lookup table</li>
        <li>No mechanism to say "I don't know"</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Why LLM Prompting Fails for Factual Questions</div>
  <div class="ppt-line"></div>

  <div style="display:flex; flex-direction:column; gap:16px; margin-top:8px;">
    <div style="display:flex; gap:16px; align-items:flex-start;">
      <div style="font-size:48px; min-width:56px;">🌀</div>
      <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 18px; background:#f7f8fc; flex:1;">
        <div style="font-size:32px; font-weight:900; color:#1f4e9a; margin-bottom:6px;">Hallucination Problem</div>
        <div style="font-size:28px; line-height:1.45;">LLMs fabricate details confidently when they don't actually know the answer.</div>
      </div>
    </div>
    <div style="display:flex; gap:16px; align-items:flex-start;">
      <div style="font-size:48px; min-width:56px;">🔒</div>
      <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 18px; background:#f7f8fc; flex:1;">
        <div style="font-size:32px; font-weight:900; color:#1f4e9a; margin-bottom:6px;">No Access to Private or Proprietary Data</div>
        <div style="font-size:28px; line-height:1.45;">Cannot answer questions about personal email, medical records, internal corporate documents, or legal materials.</div>
      </div>
    </div>
    <div style="display:flex; gap:16px; align-items:flex-start;">
      <div style="font-size:48px; min-width:56px;">📅</div>
      <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 18px; background:#f7f8fc; flex:1;">
        <div style="font-size:32px; font-weight:900; color:#1f4e9a; margin-bottom:6px;">Out-of-Date Knowledge</div>
        <div style="font-size:28px; line-height:1.45;">LLMs are static snapshots at pretraining time. They cannot answer questions about recent events (e.g., "what happened last week?").</div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Retrieval-Augmented Generation (RAG)</div>
  <div class="ppt-line"></div>

  <div style="font-size:34px; font-weight:800; margin-bottom:16px;">
    <span style="color:#1f4e9a;">Key idea:</span> use Information Retrieval to ground the LLM's answer in real documents.
  </div>

  <div style="background:#e8f0fb; border-radius:16px; padding:18px 22px; font-size:32px; line-height:1.7; margin-bottom:18px; border:2px solid #1f4e9a;">
    <b>Step 1 — Retrieve:</b> find relevant documents from a curated, proprietary, or up-to-date corpus.<br>
    <b>Step 2 — Generate:</b> feed retrieved documents as context and let the LLM produce a grounded answer.
  </div>

  <div style="font-size:28px; font-weight:800; color:#1f4e9a; margin-bottom:8px;">Advantages</div>
  <div style="display:flex; gap:12px;">
    <div style="flex:1; background:#f0faf0; border:2px solid #2a9d2a; border-radius:12px; padding:12px 14px; font-size:26px;">
      ✅ Answers grounded in <b>real, verifiable text</b>
    </div>
    <div style="flex:1; background:#f0faf0; border:2px solid #2a9d2a; border-radius:12px; padding:12px 14px; font-size:26px;">
      ✅ Provides <b>citations</b>, improving user trust
    </div>
    <div style="flex:1; background:#f0faf0; border:2px solid #2a9d2a; border-radius:12px; padding:12px 14px; font-size:26px;">
      ✅ Overcomes <b>hallucination</b>, private data, and staleness
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="muted">Motivation</li>
    <li class="active">Information Retrieval (IR)</li>
    <li class="muted">Dense Retrieval</li>
    <li class="muted">Retrieval-Augmented Generation (RAG)</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">Information Retrieval (IR)</div>
  <div class="ppt-line"></div>

  <div style="font-size:32px; line-height:1.5; margin-bottom:18px;">
    <b>IR</b> is the field of finding material (usually documents) relevant to an
    <span style="color:#1f4e9a; font-weight:900;">information need</span> from within large collections.
  </div>

  <div class="twocol" style="--left:55%;">
    <div>
      <div style="font-size:28px; font-weight:900; color:#1f4e9a; margin-bottom:10px;">Vector Space Model</div>
      <ul class="outline-bullets small">
        <li>Map queries and documents to vectors based on word counts</li>
        <li>Use <b>cosine similarity</b> between vectors to rank documents</li>
        <li>Query vector $\vec{q}$ and document vector $\vec{d}$</li>
      </ul>
      <div style="margin-top:18px; text-align:center; font-size:32px;">
        $$\text{score}(q, d) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}||\vec{d}|}$$
      </div>
    </div>
    <div>
      <div style="border:2px solid #d9deea; border-radius:14px; padding:16px; background:#f7f8fc; font-size:26px; line-height:1.55;">
        <div style="font-weight:900; color:#1f4e9a; margin-bottom:8px;">IR System Architecture</div>
        <div style="margin-bottom:6px;">① <b>Index</b> the document collection</div>
        <div style="margin-bottom:6px;">② Receive user <b>query</b></div>
        <div style="margin-bottom:6px;">③ <b>Score</b> each document against the query</div>
        <div>④ Return <b>ranked list</b> of documents</div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">TF-IDF</div>
  <div class="ppt-line"></div>

  <div style="font-size:30px; margin-bottom:14px;">
    TF-IDF is the product of two terms — capturing both local and global word importance.
  </div>

  <div style="display:flex; gap:18px; margin-bottom:16px;">
    <div style="flex:1; border:2px solid #1f4e9a; border-radius:14px; padding:14px 16px; background:#f0f4fb;">
      <div style="font-size:28px; font-weight:900; color:#1f4e9a; margin-bottom:8px;">Term Frequency (TF)</div>
      <div style="font-size:26px; margin-bottom:8px;">How frequent is word $t$ in document $d$?</div>
      <div style="text-align:center; font-size:28px;">$$\text{tf}_{t,d} = \text{count}(t, d)$$</div>
    </div>
    <div style="flex:1; border:2px solid #c00; border-radius:14px; padding:14px 16px; background:#fff8f8;">
      <div style="font-size:28px; font-weight:900; color:#c00; margin-bottom:8px;">Inverse Document Frequency (IDF)</div>
      <div style="font-size:26px; margin-bottom:8px;">Rare words are more informative:</div>
      <div style="text-align:center; font-size:28px;">$$\text{idf}_t = \log\frac{N}{\text{df}_t}$$</div>
    </div>
  </div>

  <div style="border:3px solid #1f4e9a; border-radius:14px; padding:14px 20px; background:#e8f0fb; text-align:center;">
    <div style="font-size:28px; font-weight:900; margin-bottom:6px;">TF-IDF Weight</div>
    <div style="font-size:34px;">$$w_{t,d} = \text{tf}_{t,d} \times \log\frac{N}{\text{df}_t}$$</div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">TF-IDF — Shakespeare Example</div>
  <div class="ppt-line"></div>

  <div style="font-size:28px; margin-bottom:14px;">
    Term-document matrix for four words across four Shakespeare plays:
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:26px; margin-bottom:16px;">
    <thead>
      <tr style="background:#1f4e9a; color:#fff;">
        <th style="padding:10px 14px; text-align:left;">Word</th>
        <th style="padding:10px 14px; text-align:center;">As You Like It</th>
        <th style="padding:10px 14px; text-align:center;">Twelfth Night</th>
        <th style="padding:10px 14px; text-align:center;">Julius Caesar</th>
        <th style="padding:10px 14px; text-align:center;">Henry V</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#e9eef7;">
        <td style="padding:8px 14px; font-weight:700;">battle</td>
        <td style="text-align:center;">1</td><td style="text-align:center;">0</td>
        <td style="text-align:center;">7</td><td style="text-align:center;">13</td>
      </tr>
      <tr style="background:#f7f8fc;">
        <td style="padding:8px 14px; font-weight:700;">good</td>
        <td style="text-align:center;">114</td><td style="text-align:center;">80</td>
        <td style="text-align:center;">62</td><td style="text-align:center;">89</td>
      </tr>
      <tr style="background:#e9eef7;">
        <td style="padding:8px 14px; font-weight:700;">fool</td>
        <td style="text-align:center;">36</td><td style="text-align:center;">58</td>
        <td style="text-align:center;">1</td><td style="text-align:center;">4</td>
      </tr>
      <tr style="background:#f7f8fc;">
        <td style="padding:8px 14px; font-weight:700;">wit</td>
        <td style="text-align:center;">20</td><td style="text-align:center;">15</td>
        <td style="text-align:center;">2</td><td style="text-align:center;">3</td>
      </tr>
    </tbody>
  </table>

  <div style="font-size:26px; line-height:1.5; background:#fff8f0; border:2px solid #eed9b6; border-radius:12px; padding:12px 16px;">
    💡 High-IDF words like <b>"battle"</b> (rare across plays) distinguish documents better than
    high-frequency common words like <b>"good"</b>.
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Document Scoring</div>
  <div class="ppt-line"></div>

  <div style="font-size:30px; margin-bottom:16px;">
    We score document $d$ against query $q$ using the <b>cosine of their TF-IDF vectors</b>:
  </div>

  <div style="text-align:center; font-size:36px; margin-bottom:18px;">
    $$\text{score}(q, d) = \cos(\vec{q}, \vec{d}) = \frac{\sum_{t \in q \cap d} w_{t,q} \cdot w_{t,d}}{\sqrt{\sum_t w_{t,q}^2} \cdot \sqrt{\sum_t w_{t,d}^2}}$$
  </div>

  <div class="twocol" style="--left:50%;">
    <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 16px; background:#f7f8fc; font-size:26px; line-height:1.5;">
      <div style="font-weight:900; color:#1f4e9a; margin-bottom:8px;">Properties</div>
      <ul style="padding-left:1.1em;">
        <li>Only terms appearing in <em>both</em> query and document contribute</li>
        <li>Normalized by vector lengths → length-independent</li>
        <li>Score ∈ [0, 1]</li>
      </ul>
    </div>
    <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 16px; background:#f7f8fc; font-size:26px; line-height:1.5;">
      <div style="font-weight:900; color:#1f4e9a; margin-bottom:8px;">Efficient retrieval</div>
      <ul style="padding-left:1.1em;">
        <li><b>Inverted index:</b> maps each term to its posting list</li>
        <li>Only process documents containing at least one query term</li>
        <li>Scales to billions of documents</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">BM25 — Beyond TF-IDF</div>
  <div class="ppt-line"></div>

  <div style="font-size:28px; margin-bottom:14px;">
    BM25 improves TF-IDF with <b>nonlinear TF saturation</b> and <b>length normalization</b>.
    It remains one of the strongest lexical baselines for retrieval.
  </div>

  <div style="text-align:center; font-size:28px; margin-bottom:16px; background:#e8f0fb; border-radius:12px; padding:12px;">
    $$\text{BM25}(t,d) = \frac{\text{tf}_{t,d} \cdot (k+1)}{\text{tf}_{t,d} + k\!\left(1 - b + b\,\dfrac{|d|}{\text{avgdl}}\right)} \cdot \log\frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5}$$
  </div>

  <div class="twocol" style="--left:50%;">
    <div style="border:2px solid #1f4e9a; border-radius:14px; padding:14px 16px; background:#f0f4fb; font-size:26px;">
      <div style="font-weight:900; color:#1f4e9a; margin-bottom:8px;">$k$ — TF Saturation</div>
      <ul style="padding-left:1em; line-height:1.55;">
        <li><b>Large $k$:</b> TF matters more (less saturation)</li>
        <li><b>Small $k$:</b> strong saturation, TF grows slowly</li>
        <li>Typical: $k \in [1.2, 2.0]$</li>
      </ul>
    </div>
    <div style="border:2px solid #c00; border-radius:14px; padding:14px 16px; background:#fff8f8; font-size:26px;">
      <div style="font-weight:900; color:#c00; margin-bottom:8px;">$b$ — Length Normalization</div>
      <ul style="padding-left:1em; line-height:1.55;">
        <li><b>$b = 1$:</b> full normalization by document length</li>
        <li><b>$b = 0$:</b> no length normalization</li>
        <li>Typical: $b = 0.75$</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">BM25 — Applications</div>
  <div class="ppt-line"></div>

  <div style="font-size:28px; margin-bottom:16px;">
    BM25 is the dominant <b>sparse retrieval baseline</b> in modern NLP research and production systems.
  </div>

  <div style="display:flex; flex-direction:column; gap:14px;">
    <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 18px; background:#f7f8fc; font-size:26px; line-height:1.5;">
      <div style="font-weight:900; color:#1f4e9a; margin-bottom:4px;">Adaptive RAG (COLING 2025)</div>
      <div><em>Embedding-Informed Adaptive Retrieval-Augmented Generation of Large Language Models</em></div>
      <div style="opacity:0.7; font-size:22px; margin-top:4px;">Uses BM25 as the sparse retrieval backbone in an adaptive RAG pipeline.</div>
    </div>
    <div style="border:2px solid #d9deea; border-radius:14px; padding:14px 18px; background:#f7f8fc; font-size:26px; line-height:1.5;">
      <div style="font-weight:900; color:#1f4e9a; margin-bottom:4px;">PopQA Benchmark</div>
      <div><em>When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories</em></div>
      <div style="opacity:0.7; font-size:22px; margin-top:4px;">PopQA: 14k entity-centric QA pairs. BM25 is the retrieval baseline.</div>
    </div>
    <div style="border:2px solid #2a9d2a; border-radius:14px; padding:12px 18px; background:#f0faf0; font-size:26px;">
      <div style="font-weight:900;">Tools: <a href="https://www.elastic.co" target="_blank">Elasticsearch</a> &nbsp;·&nbsp;
      <a href="https://github.com/castorini/pyserini" target="_blank">Pyserini</a> &nbsp;·&nbsp;
      <a href="https://github.com/primeqa/primeqa" target="_blank">PrimeQA</a></div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="muted">Motivation</li>
    <li class="muted">Information Retrieval (IR)</li>
    <li class="active">Dense Retrieval</li>
    <li class="muted">Retrieval-Augmented Generation (RAG)</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">From Sparse to Dense Retrieval</div>
  <div class="ppt-line"></div>

  <div style="font-size:32px; font-weight:800; color:#c00; margin-bottom:14px;">
    Vocabulary Mismatch Problem
  </div>

  <div style="font-size:28px; margin-bottom:16px; background:#fff8f8; border:2px solid #c00; border-radius:12px; padding:12px 16px; line-height:1.5;">
    BM25 and TF-IDF rely on <b>exact word overlap</b> between query and document.<br>
    Sparse vectors <em>count words</em> — no semantic understanding.
  </div>

  <div class="twocol" style="--left:50%;">
    <div style="border:2px solid #c00; border-radius:14px; padding:14px 16px; background:#fff8f8; font-size:26px; line-height:1.5;">
      <div style="font-weight:900; color:#c00; margin-bottom:8px;">❌ Sparse (BM25) fails when:</div>
      <div style="margin-bottom:6px;"><b>Query:</b> "automobile fuel efficiency"</div>
      <div style="margin-bottom:12px;"><b>Doc:</b> "car gas mileage improvement"</div>
      <div style="opacity:0.8;">Zero overlap → score = 0, despite perfect semantic match</div>
    </div>
    <div style="border:2px solid #2a9d2a; border-radius:14px; padding:14px 16px; background:#f0faf0; font-size:26px; line-height:1.5;">
      <div style="font-weight:900; color:#2a9d2a; margin-bottom:8px;">✅ Dense retrieval solves this:</div>
      <ul style="padding-left:1em;">
        <li>Encode query and document into <b>dense vectors</b></li>
        <li>Capture <b>semantic similarity</b>, not just lexical overlap</li>
        <li>Use approximate nearest-neighbor search (FAISS)</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Neural IR — Cross-Encoder</div>
  <div class="ppt-line"></div>

  <div style="font-size:30px; margin-bottom:16px;">
    Concatenate query and document, pass through BERT together:
  </div>

  <div style="text-align:center; font-size:28px; margin-bottom:18px; background:#e8f0fb; border-radius:14px; padding:16px; border:2px solid #1f4e9a;">
    $$\text{score}(q, d) = \text{BERT}([q; d])$$
  </div>

  <div class="twocol" style="--left:50%;">
    <div style="border:2px solid #2a9d2a; border-radius:14px; padding:14px 16px; background:#f0faf0; font-size:26px; line-height:1.55;">
      <div style="font-weight:900; color:#2a9d2a; margin-bottom:8px;">✅ Advantages</div>
      <ul style="padding-left:1em;">
        <li>Full interaction between every query token and every document token</li>
        <li><b>Incredibly rich</b> relevance signal</li>
        <li>Highest accuracy</li>
      </ul>
    </div>
    <div style="border:2px solid #c00; border-radius:14px; padding:14px 16px; background:#fff8f8; font-size:26px; line-height:1.55;">
      <div style="font-weight:900; color:#c00; margin-bottom:8px;">❌ Won't Scale</div>
      <ul style="padding-left:1em;">
        <li>Must run BERT on <em>every</em> (query, doc) pair at query time</li>
        <li>Cannot pre-compute document representations</li>
        <li>Infeasible for millions of documents</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Neural IR — Dense Passage Retrieval (DPR)</div>
  <div class="ppt-line"></div>

  <div style="font-size:30px; margin-bottom:16px;">
    Use <b>two separate encoders</b> — one for queries, one for documents (bi-encoder):
  </div>

  <div style="text-align:center; font-size:28px; margin-bottom:16px; background:#e8f0fb; border-radius:14px; padding:14px; border:2px solid #1f4e9a;">
    $$\text{score}(q, d) = E_Q(q)^\top E_D(d)$$
  </div>

  <div class="twocol" style="--left:50%;">
    <div style="border:2px solid #2a9d2a; border-radius:14px; padding:14px 16px; background:#f0faf0; font-size:26px; line-height:1.55;">
      <div style="font-weight:900; color:#2a9d2a; margin-bottom:8px;">✅ Highly Scalable</div>
      <ul style="padding-left:1em;">
        <li>Document embeddings are <b>pre-computed</b> and indexed offline</li>
        <li>At query time: encode query, ANN search only</li>
        <li>Scales to millions of documents (FAISS)</li>
      </ul>
    </div>
    <div style="border:2px solid #c00; border-radius:14px; padding:14px 16px; background:#fff8f8; font-size:26px; line-height:1.55;">
      <div style="font-weight:900; color:#c00; margin-bottom:8px;">❌ Limited Interaction</div>
      <ul style="padding-left:1em;">
        <li>Query and document are encoded <em>independently</em></li>
        <li>Single dot product — less expressive than cross-encoder</li>
        <li>All interaction compressed into one vector</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Cross-Encoder vs. DPR</div>
  <div class="ppt-line"></div>

  <div style="font-size:28px; margin-bottom:14px;">
    Both are trained with the same <b>shared loss function</b>:
    the negative log-likelihood of the positive passage.
  </div>

  <div style="text-align:center; font-size:28px; margin-bottom:18px; background:#e8f0fb; border-radius:12px; padding:14px; border:2px solid #1f4e9a;">
    $$\mathcal{L} = -\log \frac{e^{\text{sim}(q, d^+)}}{e^{\text{sim}(q, d^+)} + \sum_{j} e^{\text{sim}(q, d^-_j)}}$$
  </div>

  <table style="width:100%; border-collapse:collapse; font-size:26px;">
    <thead>
      <tr style="background:#1f4e9a; color:#fff;">
        <th style="padding:10px 14px; text-align:left;"></th>
        <th style="padding:10px 14px; text-align:center;">Cross-Encoder</th>
        <th style="padding:10px 14px; text-align:center;">DPR (Bi-Encoder)</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#e9eef7;">
        <td style="padding:8px 14px; font-weight:700;">Query-Doc interaction</td>
        <td style="text-align:center;">Full (token-level)</td>
        <td style="text-align:center;">Single dot product</td>
      </tr>
      <tr style="background:#f7f8fc;">
        <td style="padding:8px 14px; font-weight:700;">Pre-compute docs</td>
        <td style="text-align:center;">❌ No</td>
        <td style="text-align:center;">✅ Yes</td>
      </tr>
      <tr style="background:#e9eef7;">
        <td style="padding:8px 14px; font-weight:700;">Accuracy</td>
        <td style="text-align:center;">Higher</td>
        <td style="text-align:center;">Lower</td>
      </tr>
      <tr style="background:#f7f8fc;">
        <td style="padding:8px 14px; font-weight:700;">Scalability</td>
        <td style="text-align:center;">❌ Poor</td>
        <td style="text-align:center;">✅ Excellent</td>
      </tr>
    </tbody>
  </table>
</section>

---

<section class="ppt">
  <div class="ppt-title">Neural IR — ColBERT</div>
  <div class="ppt-line"></div>

  <div style="font-size:28px; font-weight:800; color:#1f4e9a; margin-bottom:10px;">
    Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
  </div>

  <div style="font-size:26px; line-height:1.5; background:#f7f8fc; border:2px solid #d9deea; border-radius:14px; padding:14px 18px; margin-bottom:16px;">
    ColBERT uses a <b>late interaction</b> architecture: query and document are encoded
    <em>separately</em> with BERT, then a lightweight <b>token-level MaxSim</b> matching is performed at query time.
    Document embeddings are <b>pre-computed and indexed</b>, enabling retrieval that is both highly accurate and
    orders of magnitude faster than cross-encoders.
  </div>

  <div class="twocol" style="--left:50%;">
    <div style="border:2px solid #1f4e9a; border-radius:14px; padding:12px 16px; background:#e8f0fb; font-size:26px;">
      <div style="font-weight:900; margin-bottom:6px;">MaxSim Scoring</div>
      $$S(q,d) = \sum_{i \in E_q} \max_{j \in E_d}\, E_{q_i} \cdot E_{d_j}^\top$$
    </div>
    <div style="border:2px solid #d9deea; border-radius:14px; padding:12px 16px; background:#f7f8fc; font-size:26px; line-height:1.55;">
      <div style="font-weight:900; color:#1f4e9a; margin-bottom:6px;">Best of both worlds</div>
      <ul style="padding-left:1em;">
        <li>Token-level interaction (richer than DPR)</li>
        <li>Pre-computed doc embeddings (scalable like DPR)</li>
        <li>Faster than cross-encoder by 10–100×</li>
      </ul>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">ColBERT — Late Interaction</div>
  <div class="ppt-line"></div>

  <div style="font-size:26px; margin-bottom:14px;">
    Each query token finds its <b>most similar document token</b> (MaxSim), then scores are summed:
  </div>

  <div style="background:#f7f8fc; border:2px solid #d9deea; border-radius:14px; padding:18px 20px; font-size:26px; line-height:1.7; margin-bottom:16px;">
    <div style="display:flex; gap:24px; align-items:center; justify-content:center; flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="font-weight:900; color:#1f4e9a; margin-bottom:6px;">Query tokens</div>
        <div style="display:flex; gap:8px;">
          <span style="background:#1f4e9a; color:#fff; border-radius:8px; padding:6px 12px; font-weight:700;">q₁</span>
          <span style="background:#1f4e9a; color:#fff; border-radius:8px; padding:6px 12px; font-weight:700;">q₂</span>
          <span style="background:#1f4e9a; color:#fff; border-radius:8px; padding:6px 12px; font-weight:700;">q₃</span>
        </div>
      </div>
      <div style="font-size:40px; color:#1f4e9a; font-weight:900;">→ MaxSim →</div>
      <div style="text-align:center;">
        <div style="font-weight:900; color:#c00; margin-bottom:6px;">Document tokens</div>
        <div style="display:flex; gap:8px;">
          <span style="background:#e9eef7; border-radius:8px; padding:6px 12px; font-weight:700;">d₁</span>
          <span style="background:#e9eef7; border-radius:8px; padding:6px 12px; font-weight:700;">d₂</span>
          <span style="background:#e9eef7; border-radius:8px; padding:6px 12px; font-weight:700;">d₃</span>
          <span style="background:#e9eef7; border-radius:8px; padding:6px 12px; font-weight:700;">d₄</span>
          <span style="background:#e9eef7; border-radius:8px; padding:6px 12px; font-weight:700;">d₅</span>
        </div>
      </div>
    </div>
  </div>

  <div style="font-size:26px; line-height:1.5;">
    Each query token $q_i$ is matched to the <b>single most similar</b> document token $d_j$ using dot product.
    This <em>soft alignment</em> enables fine-grained semantic comparison without running BERT on every
    query–document pair at query time.
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
    <li class="muted">Motivation</li>
    <li class="muted">Information Retrieval (IR)</li>
    <li class="muted">Dense Retrieval</li>
    <li class="active">Retrieval-Augmented Generation (RAG)</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">RAG — Full Pipeline</div>
  <div class="ppt-line"></div>

  <div style="font-size:28px; margin-bottom:16px;">
    RAG combines a <b>retriever</b> and a <b>generator</b> into an end-to-end system:
  </div>

  <div style="display:flex; flex-direction:column; gap:10px; margin-bottom:16px;">
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="background:#1f4e9a; color:#fff; border-radius:999px; width:42px; height:42px; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:22px; flex-shrink:0;">1</div>
      <div style="flex:1; border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc; font-size:26px;">
        <b>User query</b> → retrieve top-$k$ documents using DPR / BM25 / ColBERT
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="background:#1f4e9a; color:#fff; border-radius:999px; width:42px; height:42px; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:22px; flex-shrink:0;">2</div>
      <div style="flex:1; border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc; font-size:26px;">
        <b>Prepend</b> retrieved passages to the prompt as context
      </div>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
      <div style="background:#1f4e9a; color:#fff; border-radius:999px; width:42px; height:42px; display:flex; align-items:center; justify-content:center; font-weight:900; font-size:22px; flex-shrink:0;">3</div>
      <div style="flex:1; border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc; font-size:26px;">
        <b>LLM generates</b> an answer grounded in the retrieved context
      </div>
    </div>
  </div>

  <div style="background:#e8f0fb; border-radius:14px; padding:14px 18px; font-size:26px; border:2px solid #1f4e9a;">
    <div style="font-weight:900; margin-bottom:6px;">RAG Prompt Template</div>
    <div style="font-family:monospace; background:#2b2b2b; color:#eaeaea; border-radius:10px; padding:12px; font-size:22px; line-height:1.5;">
      Context: {retrieved_passage_1} ... {retrieved_passage_k}<br>
      Question: {user_query}<br>
      Answer:
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Evaluating RAG Systems</div>
  <div class="ppt-line"></div>

  <div class="twocol" style="--left:50%;">
    <div>
      <div style="font-size:28px; font-weight:900; color:#1f4e9a; margin-bottom:10px;">Retrieval Metrics</div>
      <div style="font-size:26px; line-height:1.55; background:#f7f8fc; border:2px solid #d9deea; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        <div style="margin-bottom:6px;"><b>Precision@k</b> — fraction of retrieved docs that are relevant</div>
        <div style="margin-bottom:6px;"><b>Recall@k</b> — fraction of relevant docs retrieved</div>
        <div><b>MRR / MAP</b> — ranked retrieval quality</div>
      </div>
      <div style="font-size:28px; font-weight:900; color:#1f4e9a; margin-bottom:10px;">Generation Metrics</div>
      <div style="font-size:26px; line-height:1.55; background:#f7f8fc; border:2px solid #d9deea; border-radius:12px; padding:12px 14px;">
        <div style="margin-bottom:6px;"><b>Exact Match (EM)</b></div>
        <div style="margin-bottom:6px;"><b>F1</b> over answer tokens</div>
        <div><b>LLM-as-judge</b> for open-ended answers</div>
      </div>
    </div>
    <div>
      <div style="font-size:28px; font-weight:900; color:#1f4e9a; margin-bottom:10px;">Key Benchmarks</div>
      <div style="display:flex; flex-direction:column; gap:10px; font-size:26px;">
        <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
          <b>Natural Questions (NQ)</b><br>
          <span style="font-size:22px; opacity:0.8;">100k real Google queries with Wikipedia answers</span>
        </div>
        <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
          <b>TriviaQA</b><br>
          <span style="font-size:22px; opacity:0.8;">650k trivia question-answer-evidence triples</span>
        </div>
        <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
          <b>PopQA</b><br>
          <span style="font-size:22px; opacity:0.8;">14k entity-centric QA pairs (long-tail knowledge)</span>
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">References</div>
  <div class="ppt-line"></div>

  <div style="display:flex; flex-direction:column; gap:10px; font-size:24px; line-height:1.5;">
    <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
      [1] Jurafsky & Martin. <em>Speech and Language Processing</em>, Chapter 11: Information Retrieval and RAG. Draft, Jan 2026.
    </div>
    <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
      [2] Karpukhin et al. <em>Dense Passage Retrieval for Open-Domain Question Answering.</em> EMNLP 2020.
    </div>
    <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
      [3] Lewis et al. <em>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.</em> NeurIPS 2020.
    </div>
    <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
      [4] Khattab & Zaharia. <em>ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.</em> SIGIR 2020.
    </div>
    <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
      [5] Robertson & Zaragoza. <em>The Probabilistic Relevance Framework: BM25 and Beyond.</em> 2009.
    </div>
    <div style="border:2px solid #d9deea; border-radius:12px; padding:10px 14px; background:#f7f8fc;">
      [6] Stanford CS224U Neural IR slides: <a href="https://web.stanford.edu/class/cs224u/slides/cs224u-neuralir-2023-handout.pdf" target="_blank">cs224u-neuralir-2023-handout.pdf</a>
    </div>
  </div>
</section>
