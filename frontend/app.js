const chatBody = document.getElementById("chat-body");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chatToggle = document.getElementById("chat-toggle");
const chatWidget = document.getElementById("chat-widget");

chatToggle.addEventListener("click", () => {
  chatWidget.classList.toggle("collapsed");
});

function appendMessage(role, text, sources = []) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  div.appendChild(paragraph);

  if (sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = `Sources: ${sources.join(", ")}`;
    div.appendChild(src);
  }

  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
  return div;
}

async function sendMessage(event) {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  appendMessage("user", message);
  chatInput.value = "";
  chatSend.disabled = true;
  const pending = appendMessage("bot", "Thinking...");

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    pending.querySelector("p").textContent = data.answer || "I could not answer that yet.";
    const sources = data.source_documents || [];
    if (sources.length) {
      const src = document.createElement("div");
      src.className = "sources";
      src.textContent = `Sources: ${sources.join(", ")}`;
      pending.appendChild(src);
    }
  } catch (error) {
    pending.querySelector("p").textContent =
      "The backend is unavailable. Start the API with uvicorn or Docker Compose.";
    pending.querySelector("p").classList.add("error");
  } finally {
    chatSend.disabled = false;
    chatInput.focus();
  }
}

async function loadSiteMeta() {
  const versionNode = document.getElementById("site-version");
  const yearNode = document.getElementById("site-year");

  if (yearNode) {
    yearNode.textContent = new Date().getFullYear();
  }

  try {
    const health = await fetch("/health");
    if (!health.ok) throw new Error("Health check failed");
    const data = await health.json();
    if (versionNode && data.version) {
      versionNode.textContent = `v${data.version}`;
    }
  } catch (error) {
    if (versionNode) {
      versionNode.textContent = "v0.01";
    }
  }
}

chatForm.addEventListener("submit", sendMessage);
document.addEventListener("DOMContentLoaded", loadSiteMeta);
