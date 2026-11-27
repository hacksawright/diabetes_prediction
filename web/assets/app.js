const get = id => document.getElementById(id);
const predictionForm = get('prediction-form');
const predictionButton = get('prediction-submit');
const result = get('prediction-result');
const predictionStatus = get('prediction-status');
const questionForm = get('question-form');
const questionInput = get('question');
const questionButton = get('question-submit');
const newChatButton = get('new-chat');
const questionStatus = get('question-status');
const messages = get('messages');

let predictionId = null;
let chatSessionId = null;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function setStatus(node, text, error = false) {
  node.textContent = text;
  node.classList.toggle('error', error);
}

async function post(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (typeof payload.detail === 'string') throw new Error(payload.detail);
    if (response.status === 422) throw new Error('Hãy kiểm tra lại các chỉ số và giá trị đã nhập.');
    throw new Error('Không thể xử lý yêu cầu. Hãy thử lại.');
  }
  return payload;
}

function renderSources(sources = [], includeExcerpts = false) {
  const list = element('ul', null, 'sources');
  for (const [index, source] of sources.entries()) {
    let url;
    try {
      url = source.url === '/kb/brfss' ? new URL(source.url, location.origin) : new URL(source.url);
    } catch {
      continue;
    }
    if (!['https:', 'http:'].includes(url.protocol)) continue;
    const item = element('li');
    const page = source.page ? ` · trang PDF ${source.page}` : '';
    const link = element('a', `[${source.number ?? index + 1}] ${source.title || url.hostname}${page}`);
    link.href = url.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    item.append(link);
    if (includeExcerpts && source.excerpt) item.append(element('p', source.excerpt, 'muted'));
    list.append(item);
  }
  return list;
}

function renderReport(report) {
  const { prediction, explanation, sources } = report;
  const decimal = new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
  const percent = new Intl.NumberFormat('vi-VN', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const score = element('div', null, 'score');
  score.append(
    element('span', 'Điểm tham khảo'),
    element('strong', percent.format(prediction.score)),
  );
  result.replaceChildren(score, element('p', explanation));
  if (report.rules?.review_required) {
    const review = element('div', null, 'prediction-review');
    const flags = element('ul');
    for (const flag of report.rules.flags) flags.append(element('li', flag.message));
    review.append(element('h3', 'Lưu ý khi đọc kết quả'), flags);
    result.append(review);
  }
  const metrics = prediction.metrics;
  if (metrics) {
    const details = element('details');
    details.append(
      element('summary', 'Đánh giá mô hình'),
      element('p', `BRFSS 2015 · ROC AUC ${decimal.format(metrics.roc_auc)} · Average Precision ${decimal.format(metrics.average_precision)}.`, 'muted'),
    );
    result.append(details);
  }
  const list = renderSources(sources, true);
  if (list.childElementCount) {
    const details = element('details');
    details.append(element('summary', 'Tài liệu tham khảo'), list);
    result.append(details);
  }
  const trace = report.workflow_trace ?? report.agent_trace;
  if (trace?.length) {
    const details = element('details');
    const steps = element('ol', null, 'workflow-trace');
    const states = { ok: 'Hoàn thành', degraded: 'Thiếu một phần', error: 'Lỗi' };
    const labels = {
      prepare_input: 'Chuẩn bị dữ liệu',
      model_inference: 'Chạy mô hình BRFSS',
      rule_review: 'Rà soát theo luật',
      lookup_references: 'Tìm tài liệu tham khảo',
      format_explanation: 'Tạo giải thích theo mẫu',
      assemble_report: 'Tổng hợp báo cáo',
      write_audit: 'Ghi nhật ký',
    };
    for (const step of trace) {
      const elapsed = step.duration_ms === undefined ? '' : ` · ${decimal.format(step.duration_ms)} ms`;
      const name = step.step ?? step.agent;
      steps.append(element('li', `${labels[name] ?? name} · ${states[step.status] ?? step.status}${elapsed}`));
    }
    details.append(element('summary', 'Các bước xử lý dự đoán'), steps,
      element('p', `Mã lượt chạy ${report.request_id}`, 'muted'));
    result.append(details);
  }
  result.hidden = false;
}

function appendMessage(role, content) {
  const article = element('article', null, `message ${role}`);
  article.append(element('h3', role === 'user' ? 'Bạn' : 'Trợ lý'), element('p', content));
  messages.append(article);
  return article;
}

function appendReferences(article, sources, retrieval) {
  const list = renderSources(sources);
  if (list.childElementCount) {
    const details = element('details');
    details.append(element('summary', 'Nguồn tham khảo'), list);
    article.append(details);
  }
  if (retrieval?.actions?.length) {
    const labels = { aliases: 'Tìm lại bằng từ đồng nghĩa', graph: 'Mở rộng tìm kiếm qua Neo4j', stop: 'Dừng tìm thêm' };
    const mode = retrieval.policy === 'q_learning' ? 'RL Agent' : 'Luật dự phòng';
    const details = element('details');
    details.append(element('summary', `Các bước tìm bằng chứng · ${mode}`));
    details.append(element('p', retrieval.actions.map(action => labels[action] ?? action).join(' → ')));
    article.append(details);
  }
}

async function typeReply(paragraph, answer) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    paragraph.textContent = answer;
    return;
  }
  const parts = Array.from(answer);
  const step = Math.max(2, Math.ceil(parts.length / 65));
  for (let offset = 0; offset < parts.length; offset += step) {
    paragraph.textContent += parts.slice(offset, offset + step).join('');
    if (offset + step < parts.length) {
      await new Promise(resolve => setTimeout(resolve, 16));
    }
  }
}

newChatButton.addEventListener('click', () => {
  chatSessionId = null;
  messages.replaceChildren();
  questionForm.reset();
  setStatus(questionStatus, '');
  questionInput.focus();
});

predictionForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (predictionButton.disabled) return;

  predictionButton.disabled = true;
  setStatus(predictionStatus, 'Đang tính điểm mô hình…');

  const data = Object.fromEntries(
    [...new FormData(predictionForm)].map(([key, value]) => [key, value === '' ? null : Number(value)]),
  );
  try {
    const report = await post('/predict', data);
    renderReport(report);
    predictionId = report.prediction_id;
    if (questionStatus.classList.contains('error')) setStatus(questionStatus, '');
    setStatus(predictionStatus, '');
  } catch (error) {
    setStatus(predictionStatus, error.message, true);
  } finally {
    predictionButton.disabled = false;
  }
});

questionForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (questionButton.disabled) return;
  const question = questionInput.value.trim();
  if (!question) {
    setStatus(questionStatus, 'Hãy nhập tin nhắn.', true);
    return;
  }

  questionInput.disabled = true;
  questionButton.disabled = true;
  newChatButton.disabled = true;
  const userMessage = appendMessage('user', question);
  setStatus(questionStatus, 'Đang trả lời…');
  try {
    const response = await post('/rag/ask', {
      question,
      use_chat_memory: true,
      chat_session_id: chatSessionId,
      prediction_id: predictionId,
    });
    chatSessionId = response.chat_session_id;
    setStatus(questionStatus, '');
    const assistantMessage = appendMessage('assistant', '');
    await typeReply(assistantMessage.querySelector('p'), response.answer);
    appendReferences(assistantMessage, response.sources, response.retrieval);
    questionForm.reset();
  } catch (error) {
    userMessage.remove();
    setStatus(questionStatus, error.message, true);
  } finally {
    questionInput.disabled = false;
    questionButton.disabled = false;
    newChatButton.disabled = false;
    questionInput.focus();
  }
});

