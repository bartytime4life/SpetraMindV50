// SpectraMind V50 – Small helpers for diagnostics dashboard interactivity

function toggle(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.display = (el.style.display === "none") ? "block" : "none";
}

function copyToClipboard(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const text = el.innerText || el.textContent || "";
  navigator.clipboard.writeText(text).then(() => {
    alert("Copied to clipboard.");
  }).catch(() => {
    alert("Copy failed.");
  });
}

function setHTML(id, html) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = html;
}
