#!/usr/bin/env python3
"""Build a self-contained offline dashboard for factor evaluation results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from paths import FACTOR_OUTPUT_DIR


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f6f8;
      --panel: #ffffff;
      --ink: #12212c;
      --muted: #667681;
      --line: #dbe3e8;
      --navy: #13384b;
      --cyan: #0b7c86;
      --cyan-soft: #e1f3f3;
      --green: #157347;
      --green-soft: #e6f5ed;
      --red: #b23a3a;
      --red-soft: #fbeaea;
      --amber: #a6650b;
      --shadow: 0 16px 40px rgba(27, 52, 68, 0.08);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 10% 0%, rgba(11, 124, 134, 0.09), transparent 28rem),
        var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
        "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    button, input, select { font: inherit; }
    .shell { max-width: 1680px; margin: 0 auto; padding: 28px; }
    .hero {
      position: relative;
      overflow: hidden;
      padding: 28px 30px;
      border-radius: 22px;
      color: white;
      background: linear-gradient(125deg, #102f40 0%, #164f60 63%, #0b7c86 100%);
      box-shadow: var(--shadow);
    }
    .hero::after {
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -80px;
      top: -190px;
      border: 46px solid rgba(255,255,255,0.07);
      border-radius: 50%;
    }
    .eyebrow {
      margin: 0 0 7px;
      color: #9bd9dc;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 44px); letter-spacing: -0.035em; }
    .hero-copy { max-width: 760px; margin: 10px 0 0; color: #d5e8ed; line-height: 1.65; }
    .hero-meta { margin-top: 14px; color: #a9cbd3; font-size: 12px; }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0;
    }
    .summary-card {
      min-height: 98px;
      padding: 18px;
      border: 1px solid rgba(219, 227, 232, 0.8);
      border-radius: var(--radius);
      background: rgba(255,255,255,0.92);
      box-shadow: 0 8px 24px rgba(27, 52, 68, 0.05);
    }
    .summary-label { color: var(--muted); font-size: 12px; font-weight: 700; }
    .summary-value { margin-top: 8px; font-size: 26px; font-weight: 850; letter-spacing: -0.03em; }
    .summary-note { margin-top: 3px; color: var(--muted); font-size: 11px; }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 170px 220px auto;
      gap: 12px;
      align-items: end;
      padding: 16px;
      margin-bottom: 18px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: 0 8px 24px rgba(27, 52, 68, 0.04);
    }
    .control label {
      display: block;
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .control input, .control select {
      width: 100%;
      height: 42px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      outline: none;
      color: var(--ink);
      background: #fbfcfd;
    }
    .control input:focus, .control select:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 0 3px rgba(11,124,134,0.12);
    }
    .reset {
      height: 42px;
      padding: 0 18px;
      border: 0;
      border-radius: 10px;
      color: white;
      background: var(--navy);
      cursor: pointer;
      font-weight: 750;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(580px, 0.88fr) minmax(540px, 1.12fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }
    .panel-title { margin: 0; font-size: 16px; }
    .panel-subtitle { color: var(--muted); font-size: 12px; }
    .table-wrap { max-height: 760px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    thead { position: sticky; top: 0; z-index: 2; background: #edf3f5; }
    th {
      padding: 10px 9px;
      color: #4d626e;
      text-align: right;
      white-space: nowrap;
      font-size: 10px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    th:first-child, td:first-child { text-align: left; }
    td {
      padding: 10px 9px;
      border-top: 1px solid #edf1f3;
      text-align: right;
      white-space: nowrap;
    }
    tbody tr { cursor: pointer; transition: background 120ms ease; }
    tbody tr:hover { background: #f3f9fa; }
    tbody tr.active { background: var(--cyan-soft); box-shadow: inset 3px 0 var(--cyan); }
    tbody tr:focus-visible { outline: 3px solid rgba(11,124,134,0.25); outline-offset: -3px; }
    .factor-name { color: var(--navy); font-weight: 800; }
    .metric-positive { color: var(--green); }
    .metric-negative { color: var(--red); }
    .detail { position: sticky; top: 18px; }
    .detail-content { padding: 20px; }
    .detail-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      margin-bottom: 16px;
    }
    .detail-name { margin: 0; font-size: 25px; letter-spacing: -0.025em; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
    }
    .badge.positive { color: var(--green); background: var(--green-soft); }
    .badge.negative { color: var(--red); background: var(--red-soft); }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric-card { padding: 12px; border-radius: 12px; background: #f4f7f8; }
    .metric-card .label { color: var(--muted); font-size: 10px; font-weight: 750; }
    .metric-card .value { margin-top: 5px; font-size: 17px; font-weight: 850; }
    .chart-frame {
      margin: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fbfcfd;
    }
    .chart-frame img { display: block; width: 100%; height: auto; border-radius: 8px; }
    .section-title { margin: 18px 0 8px; color: var(--muted); font-size: 11px; font-weight: 850; letter-spacing: 0.06em; text-transform: uppercase; }
    pre {
      margin: 0;
      max-height: 150px;
      overflow: auto;
      padding: 13px;
      border-radius: 12px;
      color: #d9edf0;
      background: #122b38;
      font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px 16px;
      padding: 13px;
      border-radius: 12px;
      background: #f6f8f9;
      font-size: 11px;
    }
    .meta-item span { display: block; color: var(--muted); margin-bottom: 3px; }
    .links { display: flex; flex-wrap: wrap; gap: 8px; }
    .links a {
      padding: 8px 10px;
      border: 1px solid #b9d6d9;
      border-radius: 9px;
      color: var(--cyan);
      text-decoration: none;
      font-size: 11px;
      font-weight: 750;
      background: #f1fafa;
    }
    .links a:hover { color: white; background: var(--cyan); }
    .caution {
      margin-top: 16px;
      padding: 12px 14px;
      border-left: 3px solid var(--amber);
      border-radius: 8px;
      color: #6f4b19;
      background: #fff8e9;
      font-size: 11px;
      line-height: 1.6;
    }
    .empty { padding: 42px 18px; color: var(--muted); text-align: center; }
    @media (max-width: 1180px) {
      .summary { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .workspace { grid-template-columns: 1fr; }
      .detail { position: static; }
      .table-wrap { max-height: 520px; }
    }
    @media (max-width: 760px) {
      .shell { padding: 14px; }
      .hero { padding: 22px; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .toolbar { grid-template-columns: 1fr; }
      .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .meta-grid { grid-template-columns: 1fr 1fr; }
      .table-wrap { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">Cross-sectional research dashboard</p>
      <h1>Alpha101 Factor Performance</h1>
      <p class="hero-copy">__SUBTITLE__</p>
      <div class="hero-meta">生成时间：__GENERATED_AT__ · 数据内嵌、图表离线引用</div>
    </header>

    <section class="summary" id="summary" aria-label="总体摘要"></section>

    <section class="toolbar" aria-label="筛选和排序">
      <div class="control">
        <label for="search">搜索因子或表达式</label>
        <input id="search" type="search" placeholder="例如：alpha101_044、ts_corr、volume">
      </div>
      <div class="control">
        <label for="direction">IC 方向</label>
        <select id="direction">
          <option value="all">全部方向</option>
          <option value="positive">IC &gt; 0</option>
          <option value="negative">IC &lt; 0</option>
        </select>
      </div>
      <div class="control">
        <label for="sort">排序</label>
        <select id="sort">
          <option value="ir_desc">Signed IR：高 → 低</option>
          <option value="abs_ir_desc">|IR|：高 → 低</option>
          <option value="ic_desc">IC Mean：高 → 低</option>
          <option value="gn_desc">最高组累计收益：高 → 低</option>
          <option value="name_asc">因子编号：低 → 高</option>
        </select>
      </div>
      <button class="reset" id="reset" type="button">重置视图</button>
    </section>

    <section class="workspace">
      <article class="panel">
        <div class="panel-head">
          <div>
            <h2 class="panel-title">Factor Universe</h2>
            <div class="panel-subtitle" id="result-count"></div>
          </div>
          <div class="panel-subtitle">点击行查看图表</div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Factor</th>
                <th>IC Mean</th>
                <th>IR</th>
                <th>|IR|</th>
                <th>IC+</th>
                <th>G<sub>N</sub> Final</th>
                <th>Obs</th>
              </tr>
            </thead>
            <tbody id="factor-rows"></tbody>
          </table>
          <div class="empty" id="empty" hidden>没有匹配的因子。</div>
        </div>
      </article>

      <article class="panel detail" aria-live="polite">
        <div class="detail-content" id="detail"></div>
      </article>
    </section>
  </main>

  <script>
    const FACTORS = __FACTOR_DATA__;

    const el = (id) => document.getElementById(id);
    const state = { search: "", direction: "all", sort: "ir_desc", selected: null };

    const numeric = (value) => {
      if (value === null || value === undefined || value === "") return null;
      return Number.isFinite(Number(value)) ? Number(value) : null;
    };
    const fmt = (value, digits = 4) => {
      const n = numeric(value);
      return n === null ? "—" : n.toFixed(digits);
    };
    const fmtInt = (value) => {
      const n = numeric(value);
      return n === null ? "—" : new Intl.NumberFormat("zh-CN").format(Math.round(n));
    };
    const fmtPct = (value, digits = 1) => {
      const n = numeric(value);
      return n === null ? "—" : `${(n * 100).toFixed(digits)}%`;
    };
    const fmtReturn = (value) => {
      const n = numeric(value);
      if (n === null) return "—";
      return `${new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(n * 100)}%`;
    };
    const absIr = (factor) => {
      const n = numeric(factor.ir);
      return n === null ? null : Math.abs(n);
    };
    const median = (values) => {
      const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
      if (!sorted.length) return null;
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    const metricClass = (value) => {
      const n = numeric(value);
      return n === null ? "" : (n >= 0 ? "metric-positive" : "metric-negative");
    };
    const methodSet = (factor) => new Set(String(
      factor.evaluation_methods ||
      "rank_ic,rank_icir,quantile_returns,quantile_cumulative,quantile_plot"
    ).split(",").filter(Boolean));
    const parsePathMap = (value) => {
      if (!value || typeof value !== "string") return {};
      try {
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
      } catch {
        return {};
      }
    };
    const escapeHtml = (value) => String(value ?? "")
      .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");

    function renderSummary() {
      const ranked = FACTORS.filter((factor) => numeric(factor.ir) !== null)
        .sort((a, b) => numeric(b.ir) - numeric(a.ir));
      const best = ranked[0] || null;
      const positive = FACTORS.filter((f) => numeric(f.ic_mean) > 0).length;
      const cards = [
        ["因子数量", fmtInt(FACTORS.length), "当前结果集合"],
        ["Median |IR|", fmt(median(FACTORS.map(absIr))), "全体因子中位数"],
        ["最高 Signed IR", best ? fmt(best.ir) : "—", best ? best.factor_name : "未评价"],
        ["正 IC 因子", `${positive} / ${FACTORS.length}`, fmtPct(positive / FACTORS.length)],
        ["Median IC Obs", fmtInt(median(FACTORS.map((f) => f.ic_count))), "有效交易日"],
      ];
      el("summary").innerHTML = cards.map(([label, value, note]) => `
        <div class="summary-card">
          <div class="summary-label">${label}</div>
          <div class="summary-value">${value}</div>
          <div class="summary-note">${note}</div>
        </div>`).join("");
    }

    function filteredFactors() {
      const query = state.search.trim().toLowerCase();
      let rows = FACTORS.filter((factor) => {
        const matchesSearch = !query ||
          factor.factor_name.toLowerCase().includes(query) ||
          factor.expression.toLowerCase().includes(query);
        const matchesDirection =
          state.direction === "all" ||
          (state.direction === "positive" && factor.ic_mean > 0) ||
          (state.direction === "negative" && factor.ic_mean < 0);
        return matchesSearch && matchesDirection;
      });
      const sorters = {
        ir_desc: (a, b) => b.ir - a.ir,
        abs_ir_desc: (a, b) => (absIr(b) ?? -Infinity) - (absIr(a) ?? -Infinity),
        ic_desc: (a, b) => b.ic_mean - a.ic_mean,
        gn_desc: (a, b) => b.gn_final_cumulative - a.gn_final_cumulative,
        name_asc: (a, b) => a.factor_name.localeCompare(b.factor_name),
      };
      return rows.sort(sorters[state.sort]);
    }

    function selectFactor(name, updateHash = true) {
      const factor = FACTORS.find((row) => row.factor_name === name);
      if (!factor) return;
      state.selected = name;
      if (updateHash) history.replaceState(null, "", `#${name}`);
      renderTable();
      renderDetail(factor);
    }

    function renderTable() {
      const rows = filteredFactors();
      el("result-count").textContent = `显示 ${rows.length} / ${FACTORS.length}`;
      el("empty").hidden = rows.length > 0;
      el("factor-rows").innerHTML = rows.map((factor) => `
        <tr tabindex="0" data-factor="${factor.factor_name}" class="${state.selected === factor.factor_name ? "active" : ""}">
          <td class="factor-name">${factor.factor_name}</td>
          <td class="${metricClass(factor.ic_mean)}">${fmt(factor.ic_mean)}</td>
          <td class="${metricClass(factor.ir)}">${fmt(factor.ir)}</td>
          <td>${fmt(absIr(factor))}</td>
          <td>${fmtPct(factor.ic_positive_ratio, 0)}</td>
          <td class="${metricClass(factor.gn_final_cumulative)}">${fmtReturn(factor.gn_final_cumulative)}</td>
          <td>${fmtInt(factor.ic_count)}</td>
        </tr>`).join("");
      el("factor-rows").querySelectorAll("tr").forEach((row) => {
        row.addEventListener("click", () => selectFactor(row.dataset.factor));
        row.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            selectFactor(row.dataset.factor);
          }
        });
      });
    }

    function metricCard(label, value, className = "") {
      return `<div class="metric-card"><div class="label">${label}</div><div class="value ${className}">${value}</div></div>`;
    }

    function renderDetail(factor) {
      const methods = methodSet(factor);
      const icMean = numeric(factor.ic_mean);
      const direction = icMean === null ? "" : (icMean >= 0 ? "positive" : "negative");
      const directionLabel = icMean === null ? "未评价 IC" : (icMean >= 0 ? "正向 IC" : "反向 IC");
      const artifact = encodeURIComponent(factor.artifact_name);
      let detailPaths = parsePathMap(factor.evaluation_details);
      let artifactPaths = parsePathMap(factor.evaluation_artifacts);
      if (!Object.keys(detailPaths).length && !Object.keys(artifactPaths).length) {
        if (methods.has("rank_ic")) detailPaths.ic = `details/${artifact}__ic.csv`;
        if (methods.has("quantile_returns")) detailPaths.group_returns = `details/${artifact}__group_returns.csv`;
        if (methods.has("quantile_cumulative")) detailPaths.cumulative_returns = `details/${artifact}__cumulative_returns.csv`;
        if (methods.has("quantile_plot")) artifactPaths.plot = `plots/${artifact}.png`;
      }
      const charts = Object.entries(artifactPaths)
        .filter(([, path]) => /\.(png|jpe?g|gif|webp|svg)$/i.test(path))
        .map(([name, path]) => `
          <figure class="chart-frame">
            <img src="${encodeURI(path)}" alt="${escapeHtml(factor.factor_name)} ${escapeHtml(name)}">
          </figure>`).join("");
      const linkLabels = {
        ic: "日度 IC",
        group_returns: "分组收益",
        cumulative_returns: "累计收益",
        plot: "分位数组合图",
        rolling_sharpe_60_plot: "60日非重叠夏普图",
      };
      const links = [`<a href="code/${artifact}.py">复跑代码</a>`];
      Object.entries(detailPaths).forEach(([name, path]) => {
        links.push(`<a href="${encodeURI(path)}">${escapeHtml(linkLabels[name] || name)}</a>`);
      });
      Object.entries(artifactPaths).forEach(([name, path]) => {
        links.push(`<a href="${encodeURI(path)}">${escapeHtml(linkLabels[name] || name)}</a>`);
      });
      const reservedMetrics = new Set([
        "run_id", "evaluated_at", "factor_name", "artifact_name", "expression",
        "horizon", "return_definition", "n_quantiles", "evaluation_methods", "evaluation_details",
        "evaluation_artifacts", "ic_mean", "ic_std", "ir", "abs_ir",
        "ic_positive_ratio", "ic_count", "pair_count", "start_day", "end_day",
        "g1_final_cumulative", "gn_final_cumulative",
      ]);
      const additionalMetrics = Object.entries(factor)
        .filter(([name, value]) => !reservedMetrics.has(name) && numeric(value) !== null)
        .map(([name, value]) => metricCard(escapeHtml(name), fmt(value)))
        .join("");
      el("detail").innerHTML = `
        <div class="detail-top">
          <div>
            <div class="eyebrow" style="color: var(--cyan)">Selected factor</div>
            <h2 class="detail-name">${factor.factor_name}</h2>
          </div>
          <span class="badge ${direction}">${directionLabel}</span>
        </div>
        <div class="metrics-grid">
          ${metricCard("IC Mean", fmt(factor.ic_mean), metricClass(factor.ic_mean))}
          ${metricCard("IC Std", fmt(factor.ic_std))}
          ${metricCard("Signed IR", fmt(factor.ir), metricClass(factor.ir))}
          ${metricCard("|IR|", fmt(absIr(factor)))}
          ${metricCard("IC Positive", fmtPct(factor.ic_positive_ratio))}
          ${metricCard("IC Obs", fmtInt(factor.ic_count))}
          ${metricCard(`G${factor.n_quantiles} Final`, fmtReturn(factor.gn_final_cumulative), metricClass(factor.gn_final_cumulative))}
          ${metricCard(`G${factor.n_quantiles} Final`, fmtReturn(factor.gn_final_cumulative), metricClass(factor.gn_final_cumulative))}
          ${additionalMetrics}
        </div>
        ${charts}
        <div class="section-title">Expression</div>
        <pre>${escapeHtml(factor.expression)}</pre>
        <div class="section-title">Evaluation metadata</div>
        <div class="meta-grid">
          <div class="meta-item"><span>Period</span>${escapeHtml(factor.start_day)} → ${escapeHtml(factor.end_day)}</div>
          <div class="meta-item"><span>Return label</span>${escapeHtml(factor.return_definition || "legacy / unspecified")}</div>
          <div class="meta-item"><span>Holding horizon</span>${factor.horizon} day</div>
          <div class="meta-item"><span>Quantiles</span>${factor.n_quantiles}</div>
          <div class="meta-item"><span>Pair count</span>${fmtInt(factor.pair_count)}</div>
          <div class="meta-item"><span>G1 final</span>${fmtReturn(factor.g1_final_cumulative)}</div>
          <div class="meta-item"><span>Evaluated</span>${escapeHtml(factor.evaluated_at)}</div>
          <div class="meta-item"><span>Methods</span>${escapeHtml([...methods].join(", "))}</div>
        </div>
        <div class="section-title">Artifacts</div>
        <div class="links">${links.join("")}</div>
        <div class="caution">IR 只反映 IC 均值与波动的比值。筛选因子时还应检查 IC 方向、分组单调性、头部组合表现、样本覆盖及经济逻辑。</div>`;
    }

    function bindControls() {
      el("search").addEventListener("input", (event) => {
        state.search = event.target.value;
        renderTable();
      });
      el("direction").addEventListener("change", (event) => {
        state.direction = event.target.value;
        renderTable();
      });
      el("sort").addEventListener("change", (event) => {
        state.sort = event.target.value;
        renderTable();
      });
      el("reset").addEventListener("click", () => {
        state.search = "";
        state.direction = "all";
        state.sort = "ir_desc";
        el("search").value = "";
        el("direction").value = "all";
        el("sort").value = "ir_desc";
        const first = filteredFactors()[0];
        if (first) selectFactor(first.factor_name);
      });
    }

    renderSummary();
    bindControls();
    const hashFactor = decodeURIComponent(location.hash.replace(/^#/, ""));
    const initial = FACTORS.find((factor) => factor.factor_name === hashFactor) ||
      [...FACTORS].sort(
        (a, b) => (numeric(b.ir) ?? -Infinity) - (numeric(a.ir) ?? -Infinity)
      )[0];
    selectFactor(initial.factor_name, false);
  </script>
</body>
</html>
"""

DEFAULT_SUBTITLE = (
    "集中查看当前因子集合的日度 Spearman IC、IR、十组累计收益与归档图。"
    "点击任意因子查看完整表现和复跑入口。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=FACTOR_OUTPUT_DIR / "alpha101_exact",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="Alpha101 Factor Performance")
    parser.add_argument("--subtitle", default=DEFAULT_SUBTITLE)
    return parser


def validate_artifacts(results_dir: Path, metrics: pd.DataFrame) -> None:
    missing: list[str] = []
    results_root = results_dir.resolve()
    default_methods = {
        "rank_ic",
        "rank_icir",
        "quantile_returns",
        "quantile_cumulative",
        "quantile_plot",
    }
    for row in metrics.itertuples(index=False):
        artifact = str(row.artifact_name)
        raw_methods = getattr(row, "evaluation_methods", None)
        methods = (
            {
                part.strip()
                for part in str(raw_methods).split(",")
                if part.strip()
            }
            if raw_methods is not None and not pd.isna(raw_methods)
            else default_methods
        )
        expected = [results_dir / "code" / f"{artifact}.py"]
        metadata_paths: list[str] = []
        for column in ("evaluation_details", "evaluation_artifacts"):
            raw_value = getattr(row, column, None)
            if raw_value is None or pd.isna(raw_value):
                continue
            parsed = json.loads(str(raw_value))
            if not isinstance(parsed, dict) or not all(
                isinstance(path, str) for path in parsed.values()
            ):
                raise ValueError(f"{column} must be a JSON object of relative paths")
            metadata_paths.extend(parsed.values())
        if metadata_paths:
            for relative_path in metadata_paths:
                candidate = (results_dir / relative_path).resolve()
                if candidate != results_root and results_root not in candidate.parents:
                    raise ValueError(
                        f"Evaluation artifact escapes results directory: {relative_path}"
                    )
                expected.append(candidate)
        else:
            if "rank_ic" in methods:
                expected.append(results_dir / "details" / f"{artifact}__ic.csv")
            if "quantile_returns" in methods:
                expected.append(
                    results_dir / "details" / f"{artifact}__group_returns.csv"
                )
            if "quantile_cumulative" in methods:
                expected.append(
                    results_dir / "details" / f"{artifact}__cumulative_returns.csv"
                )
            if "quantile_plot" in methods:
                expected.append(results_dir / "plots" / f"{artifact}.png")
        missing.extend(str(path) for path in expected if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Dashboard artifacts are missing: {missing[:10]}")


def build_dashboard(
    results_dir: str | Path,
    *,
    output: str | Path | None = None,
    title: str = "Alpha101 Factor Performance",
    subtitle: str = DEFAULT_SUBTITLE,
) -> dict[str, object]:
    """Build and validate one self-contained factor-results dashboard."""

    results_dir = Path(results_dir).expanduser().resolve()
    metrics_path = results_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path).sort_values("factor_name")
    if metrics.empty or metrics["factor_name"].duplicated().any():
        raise ValueError("metrics.csv must contain one nonempty row per factor")
    validate_artifacts(results_dir, metrics)

    records = json.loads(metrics.to_json(orient="records"))
    factor_json = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    html = (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__GENERATED_AT__", generated_at)
        .replace("__FACTOR_DATA__", factor_json)
    )
    output = (
        Path(output).expanduser().resolve()
        if output
        else results_dir / "index.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return {
        "output": str(output),
        "factor_count": len(metrics),
        "bytes": output.stat().st_size,
    }


def main() -> int:
    args = build_parser().parse_args()
    result = build_dashboard(
        args.results_dir,
        output=args.output,
        title=args.title,
        subtitle=args.subtitle,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
