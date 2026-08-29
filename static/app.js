const form = document.querySelector("#recommendation-form");
const statusMessage = document.querySelector("#status");
const results = document.querySelector("#results");
const submitButton = form.querySelector("button");

function addItems(containerId, items) {
  const container = document.querySelector(`#${containerId}`);
  container.replaceChildren();
  if (!items.length) {
    container.append(document.querySelector("#empty-template").content.cloneNode(true));
    return;
  }

  items.forEach((item) => {
    const element = document.createElement("li");
    const text = document.createElement("span");
    text.className = "recommendation-text";
    text.textContent = item.text;
    element.append(text);

    const provenance = document.createElement("div");
    provenance.className = "provenance";
    const label = document.createElement("span");
    label.className = "provenance-label";
    label.textContent = `Source${item.source_count > 1 ? "s" : ""} (${item.source_count})`;
    provenance.append(label);

    item.sources.forEach((source) => {
      const sourceElement = document.createElement("span");
      sourceElement.className = "source-chip";
      sourceElement.textContent = `${source.title || `RETEX #${source.retex_id ?? "?"}`} · ${source.similarity_score.toFixed(3)}`;
      provenance.append(sourceElement);
    });

    element.append(provenance);
    container.append(element);
  });
}

function displayCases(cases) {
  const container = document.querySelector("#cases");
  container.replaceChildren();

  cases.forEach((item) => {
    const card = document.createElement("article");
    card.className = "case";
    const title = item.title || "RETEX sans titre";

    card.innerHTML = `
      <span class="score">Similarité ${item.similarity_score.toFixed(3)}</span>
      <h3></h3>
      <p></p>
      <details>
        <summary>Voir les réponses apportées</summary>
        <p><strong>Actions :</strong> </p>
        <p><strong>Recommandations :</strong> </p>
      </details>
    `;

    card.querySelector("h3").textContent = title;
    card.querySelector("h3 + p").textContent = item.organization || "Organisation non renseignée";

    const details = card.querySelectorAll("details p");
    details[0].append(item.actions || "Non renseignées");
    details[1].append(item.recommendations || "Non renseignées");

    const analysis = document.createElement("div");
    analysis.className = "case-analysis";
    analysis.innerHTML = `
      <p><strong>Trigger :</strong> ${item.trigger || "Non documenté"}</p>
      <p><strong>Effet domino :</strong> ${item.domino_effect || "Non documenté"}</p>
      <p><strong>Faute structurelle :</strong> ${item.structural_failure || "Non documentée"}</p>
    `;

    card.append(analysis);
    container.append(card);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const description = document.querySelector("#description").value.trim();
  statusMessage.className = "status";
  statusMessage.textContent = "Analyse sémantique des RETEX en cours…";
  submitButton.disabled = true;

  try {
    const response = await fetch("/api/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        description,
        top_k: Number(document.querySelector("#top-k").value),
        limit: 10,
      }),
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Une erreur est survenue.");

    addItems("actions", data.actions);
    addItems("recommendations", data.recommendations);
    displayCases(data.similar_cases);
    results.hidden = false;
    statusMessage.textContent = `${data.similar_cases.length} RETEX similaires analysés.`;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    statusMessage.className = "status error";
    statusMessage.textContent = error.message;
  } finally {
    submitButton.disabled = false;
  }
});
