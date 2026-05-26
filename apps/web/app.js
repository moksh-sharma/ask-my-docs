const messagesEl = document.getElementById("messages");
const sourcesListEl = document.getElementById("sources-list");
const sourcesEmptyEl = document.getElementById("sources-empty");
const askForm = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const askBtn = document.getElementById("ask-btn");
const ingestBtn = document.getElementById("ingest-btn");
const statusPill = document.getElementById("status-pill");
const toastEl = document.getElementById("toast");

let busy = false;

function showToast(message, durationMs = 3200) {
  toastEl.textContent = message;
  toastEl.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => {
    toastEl.hidden = true;
  }, durationMs);
}

function setBusy(nextBusy) {
  busy = nextBusy;
  askBtn.disabled = nextBusy;
  ingestBtn.disabled = nextBusy;
  questionInput.disabled = nextBusy;
}

function clearWelcome() {
  const welcome = messagesEl.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function appendMessage(role, text, extraClass = "") {
  clearWelcome();
  const el = document.createElement("div");
  el.className = `message message-${role} ${extraClass}`.trim();
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function formatAnswerWithCitations(answer) {
  return answer.replace(/\[(\d+)\]/g, '<a class="citation-ref" href="#source-$1">[$1]</a>');
}

function renderSources(sources, citations) {
  if (!sources?.length) {
    sourcesListEl.hidden = true;
    sourcesEmptyEl.hidden = false;
    sourcesEmptyEl.textContent = "No sources returned.";
    return;
  }

  sourcesEmptyEl.hidden = true;
  sourcesListEl.hidden = false;
  sourcesListEl.innerHTML = "";

  const quoteById = Object.fromEntries(
    (citations || []).map((c) => [c.id, c.quote])
  );

  for (const src of sources) {
    const card = document.createElement("article");
    card.className = "source-card";
    card.id = `source-${src.index}`;
    const section = src.section ? ` · ${src.section}` : "";
    const quote = quoteById[src.index];
    card.innerHTML = `
      <strong>[${src.index}] ${escapeHtml(src.title || src.source)}</strong>
      <div class="meta">${escapeHtml(src.source)}${escapeHtml(section)}</div>
      ${quote ? `<blockquote>${escapeHtml(quote)}</blockquote>` : ""}
    `;
    sourcesListEl.appendChild(card);
  }
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail ?? res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

async function checkHealth() {
  try {
    await api("/health");
    statusPill.textContent = "API online";
    statusPill.className = "pill pill-ok";
  } catch {
    statusPill.textContent = "API offline";
    statusPill.className = "pill pill-warn";
  }
}

async function runIngest() {
  setBusy(true);
  try {
    const result = await api("/ingest?docs_dir=data/sample_docs", { method: "POST" });
    showToast(`Indexed ${result.documents} docs (${result.chunks} chunks)`);
    statusPill.textContent = "Index ready";
    statusPill.className = "pill pill-ok";
  } catch (err) {
    showToast(`Ingest failed: ${err.message}`);
  } finally {
    setBusy(false);
  }
}

async function runAsk(question) {
  const trimmed = question.trim();
  if (!trimmed || busy) return;

  setBusy(true);
  appendMessage("user", trimmed);
  questionInput.value = "";

  const loadingEl = appendMessage("assistant", "Searching docs and generating answer…", "message-loading");

  try {
    const response = await api("/ask", {
      method: "POST",
      body: JSON.stringify({ question: trimmed, include_debug: false }),
    });

    loadingEl.remove();
    const answerEl = appendMessage("assistant", "");
    answerEl.innerHTML = formatAnswerWithCitations(response.answer);
    renderSources(response.sources, response.citations);
  } catch (err) {
    loadingEl.remove();
    appendMessage("error", err.message, "message-error");
    showToast(err.message);
  } finally {
    setBusy(false);
    questionInput.focus();
  }
}

askForm.addEventListener("submit", (e) => {
  e.preventDefault();
  runAsk(questionInput.value);
});

ingestBtn.addEventListener("click", () => runIngest());

document.querySelectorAll(".suggestions button").forEach((btn) => {
  btn.addEventListener("click", () => {
    const q = btn.dataset.question;
    questionInput.value = q;
    runAsk(q);
  });
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});

checkHealth();
runIngest();
