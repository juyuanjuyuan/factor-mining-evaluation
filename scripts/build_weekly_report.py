#!/usr/bin/env python3
"""Build the factor-evaluation-system weekly work report as a PDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "outputs" / "factor_evaluation" / "alpha101_funnel_v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "pdf"
    / "factor_evaluation_weekly_report_2026-07-03.pdf"
)

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

NAVY = HexColor("#102A43")
BLUE = HexColor("#1F5A94")
TEAL = HexColor("#13A89E")
TEAL_LIGHT = HexColor("#E7F7F5")
AMBER = HexColor("#F4B740")
AMBER_LIGHT = HexColor("#FFF6DD")
RED = HexColor("#D65A4A")
RED_LIGHT = HexColor("#FBEDEA")
INK = HexColor("#243B53")
MUTED = HexColor("#627D98")
LINE = HexColor("#D9E2EC")
PANEL = HexColor("#F5F7FA")
WHITE = colors.white


def register_fonts() -> None:
    pdfmetrics.registerFont(
        TTFont(
            "STSong-Light",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        )
    )


def style(
    name: str,
    *,
    size: float,
    leading: float | None = None,
    color: colors.Color = INK,
    align: int = TA_LEFT,
    space_after: float = 0,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName="STSong-Light",
        fontSize=size,
        leading=leading or size * 1.45,
        textColor=color,
        alignment=align,
        spaceAfter=space_after,
        wordWrap="CJK",
    )


BODY = style("body", size=9.2, leading=14.2)
BODY_SMALL = style("body_small", size=8.2, leading=12.2)
CAPTION = style("caption", size=7.5, leading=10.5, color=MUTED)
SECTION = style("section", size=18, leading=24, color=NAVY)
SUBHEAD = style("subhead", size=11.5, leading=16, color=BLUE)
CARD_TITLE = style("card_title", size=10.5, leading=14, color=NAVY)
WHITE_SMALL = style("white_small", size=8.5, leading=12, color=WHITE)


def draw_paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    paragraph_style: ParagraphStyle = BODY,
    *,
    max_height: float = 200 * mm,
) -> float:
    paragraph = Paragraph(text, paragraph_style)
    _, height = paragraph.wrap(width, max_height)
    paragraph.drawOn(c, x, top - height)
    return height


def rounded_panel(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: colors.Color = WHITE,
    stroke: colors.Color = LINE,
    radius: float = 4 * mm,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.7)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def page_header(
    c: canvas.Canvas,
    page_number: int,
    title: str,
    kicker: str,
) -> None:
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 16 * mm, PAGE_W, 16 * mm, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, PAGE_H - 17.4 * mm, PAGE_W, 1.4 * mm, fill=1, stroke=0)
    c.setFont("STSong-Light", 8.5)
    c.setFillColor(WHITE)
    c.drawString(MARGIN_X, PAGE_H - 10.5 * mm, kicker)
    c.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 10.5 * mm, title)
    c.setStrokeColor(LINE)
    c.line(MARGIN_X, 13 * mm, PAGE_W - MARGIN_X, 13 * mm)
    c.setFont("STSong-Light", 7.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN_X, 8 * mm, "因子评价体系搭建工作周报")
    c.drawRightString(PAGE_W - MARGIN_X, 8 * mm, f"{page_number:02d}")


def section_title(
    c: canvas.Canvas,
    number: str,
    title: str,
    subtitle: str,
) -> float:
    top = PAGE_H - 29 * mm
    c.setFillColor(TEAL)
    c.roundRect(MARGIN_X, top - 7 * mm, 12 * mm, 7 * mm, 2 * mm, fill=1, stroke=0)
    c.setFont("STSong-Light", 8.5)
    c.setFillColor(WHITE)
    c.drawCentredString(MARGIN_X + 6 * mm, top - 4.9 * mm, number)
    draw_paragraph(c, title, MARGIN_X + 16 * mm, top + 1 * mm, CONTENT_W - 16 * mm, SECTION)
    draw_paragraph(
        c,
        subtitle,
        MARGIN_X + 16 * mm,
        top - 8 * mm,
        CONTENT_W - 16 * mm,
        CAPTION,
    )
    return top - 19 * mm


def metric_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    value: str,
    label: str,
    *,
    accent: colors.Color = TEAL,
) -> None:
    rounded_panel(c, x, y, width, 24 * mm)
    c.setFillColor(accent)
    c.roundRect(x, y, 3 * mm, 24 * mm, 2 * mm, fill=1, stroke=0)
    c.setFont("STSong-Light", 20)
    c.setFillColor(NAVY)
    c.drawString(x + 8 * mm, y + 11.5 * mm, value)
    c.setFont("STSong-Light", 8)
    c.setFillColor(MUTED)
    c.drawString(x + 8 * mm, y + 5.5 * mm, label)


def load_report_data(results_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(results_dir / "funnel_summary.csv")
    stage2 = pd.read_csv(results_dir / "stage2_ic" / "metrics.csv")
    stage2b = pd.read_csv(results_dir / "stage2b_neutral" / "metrics.csv")
    stage3 = pd.read_csv(results_dir / "stage3_portfolio" / "metrics.csv")
    status = pd.read_csv(results_dir / "funnel_status.csv")

    evaluated = status.loc[status["execution_state"].eq("evaluated")].copy()
    latest = evaluated.groupby(["factor_name", "stage"], sort=False).tail(1)
    timing_rows: list[dict[str, object]] = []
    for stage_name, rows in latest.groupby("stage", sort=False):
        slowest = rows.loc[rows["elapsed_seconds"].idxmax()]
        timing_rows.append(
            {
                "stage": stage_name,
                "count": len(rows),
                "total_minutes": float(rows["elapsed_seconds"].sum() / 60),
                "median_seconds": float(rows["elapsed_seconds"].median()),
                "p95_seconds": float(rows["elapsed_seconds"].quantile(0.95)),
                "slowest_factor": str(slowest["factor_name"]),
                "slowest_seconds": float(slowest["elapsed_seconds"]),
            }
        )
    timings = pd.DataFrame(timing_rows).set_index("stage")

    eliminated = summary.loc[summary["final_status"].eq("eliminated")].copy()
    return {
        "summary": summary,
        "stage2": stage2,
        "stage2b": stage2b,
        "stage3": stage3,
        "timings": timings,
        "eliminated": eliminated,
    }


def cover_page(c: canvas.Canvas, data: dict[str, object]) -> None:
    summary = data["summary"]
    assert isinstance(summary, pd.DataFrame)
    completed = int(summary["final_status"].eq("completed").sum())
    eliminated = int(summary["final_status"].eq("eliminated").sum())

    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, fill=1, stroke=0)
    c.setFillColor(HexColor("#173F5F"))
    c.circle(PAGE_W - 22 * mm, PAGE_H - 35 * mm, 42 * mm, fill=1, stroke=0)
    c.setFillColor(HexColor("#1C4D70"))
    c.circle(PAGE_W - 6 * mm, PAGE_H - 69 * mm, 34 * mm, fill=1, stroke=0)

    c.setFillColor(TEAL)
    c.roundRect(MARGIN_X, PAGE_H - 49 * mm, 45 * mm, 8 * mm, 2 * mm, fill=1, stroke=0)
    c.setFont("STSong-Light", 9)
    c.setFillColor(WHITE)
    c.drawCentredString(MARGIN_X + 22.5 * mm, PAGE_H - 46.5 * mm, "WEEKLY REPORT")

    c.setFont("STSong-Light", 29)
    c.setFillColor(WHITE)
    c.drawString(MARGIN_X, PAGE_H - 78 * mm, "因子评价体系搭建")
    c.setFont("STSong-Light", 24)
    c.drawString(MARGIN_X, PAGE_H - 92 * mm, "工作周报")
    c.setFont("STSong-Light", 10)
    c.setFillColor(HexColor("#B8D7E8"))
    c.drawString(
        MARGIN_X,
        PAGE_H - 106 * mm,
        "模块化评价框架 / 四阶段评价漏斗 / Alpha101 全量验证",
    )

    panel_y = PAGE_H - 183 * mm
    rounded_panel(
        c,
        MARGIN_X,
        panel_y,
        CONTENT_W,
        54 * mm,
        fill=HexColor("#173F5F"),
        stroke=HexColor("#2C5F80"),
    )
    card_w = (CONTENT_W - 24 * mm) / 3
    metrics = [
        ("83", "可运行 Alpha101 因子"),
        (str(completed), "完成组合层面评价"),
        (str(eliminated), "统计筛选淘汰"),
    ]
    for index, (value, label) in enumerate(metrics):
        x = MARGIN_X + 8 * mm + index * (card_w + 4 * mm)
        c.setFont("STSong-Light", 23)
        c.setFillColor(TEAL if index != 2 else AMBER)
        c.drawString(x, panel_y + 29 * mm, value)
        c.setFont("STSong-Light", 8.2)
        c.setFillColor(WHITE)
        c.drawString(x, panel_y + 20 * mm, label)
        c.setFillColor(HexColor("#B8D7E8"))
        c.setFont("STSong-Light", 7)
        note = ["全量进入正确性门槛", "产出图表与滚动风险明细", "3 原始 IC + 1 中性 IC"][index]
        c.drawString(x, panel_y + 13 * mm, note)

    draw_paragraph(
        c,
        "本周完成因子评价体系 1.0 的设计、开发与真实数据验证。系统以注册式 evaluator "
        "为核心，将正确性、统计显著性、风险暴露控制和组合确认拆分为可组合模块，并通过"
        "可恢复漏斗完成 83 条 Alpha101 因子的全量运行。",
        MARGIN_X,
        panel_y - 14 * mm,
        CONTENT_W,
        style("cover_summary", size=10, leading=16, color=HexColor("#D5E5EF")),
    )

    c.setFont("STSong-Light", 8)
    c.setFillColor(HexColor("#B8D7E8"))
    c.drawString(MARGIN_X, 22 * mm, "汇报周期：2026-06-29 - 2026-07-03")
    c.drawRightString(PAGE_W - MARGIN_X, 22 * mm, "版本：Factor Evaluation v1.0")
    c.showPage()


def overview_page(c: canvas.Canvas, data: dict[str, object]) -> None:
    page_header(c, 2, "本周工作概览", "01 / EXECUTIVE SUMMARY")
    top = section_title(
        c,
        "01",
        "本周完成了什么",
        "从评价函数、流程编排到结果归档，形成可复用、可恢复、可审计的评价闭环。",
    )

    card_gap = 5 * mm
    card_w = (CONTENT_W - 2 * card_gap) / 3
    cards = [
        (
            "模块化评价内核",
            "每个方法以注册函数接收统一 EvaluationState，独立贡献 metrics、details 与 artifacts；依赖由 registry 自动补齐。",
            TEAL,
        ),
        (
            "四阶段评价漏斗",
            "先检查未来函数，再做原始 IC 显著性、市值中性复测，最后运行可交易性过滤与组合评价；淘汰即停止。",
            BLUE,
        ),
        (
            "可恢复批量运行",
            "表达式、收益定义、horizon、分组数、方法列表和产物均匹配时复用结果；本次全量恢复验证仅约 2.4 秒。",
            AMBER,
        ),
    ]
    y = top - 51 * mm
    for index, (title, text, accent) in enumerate(cards):
        x = MARGIN_X + index * (card_w + card_gap)
        rounded_panel(c, x, y, card_w, 47 * mm)
        c.setFillColor(accent)
        c.roundRect(x + 6 * mm, y + 36 * mm, 12 * mm, 5 * mm, 1.5 * mm, fill=1, stroke=0)
        draw_paragraph(c, title, x + 6 * mm, y + 33 * mm, card_w - 12 * mm, CARD_TITLE)
        draw_paragraph(c, text, x + 6 * mm, y + 24 * mm, card_w - 12 * mm, BODY_SMALL)

    y2 = y - 67 * mm
    rounded_panel(c, MARGIN_X, y2, CONTENT_W, 57 * mm, fill=PANEL)
    draw_paragraph(c, "核心设计原则", MARGIN_X + 8 * mm, y2 + 49 * mm, 55 * mm, SUBHEAD)
    principles = [
        ("统一口径", "收益标签只在 returns.py 定义，默认 open[t+2]/open[t+1]-1。"),
        ("显式依赖", "方法声明 requires 与 required_data_symbols，编排层不硬编码指标字段。"),
        ("分阶段归档", "每个阶段独立 metrics.csv，避免按因子名 upsert 相互覆盖。"),
        ("证据可追溯", "保留日度 IC、分组收益、风险序列、图表和可重跑代码。"),
    ]
    row_top = y2 + 38 * mm
    for index, (title, text) in enumerate(principles):
        row = index // 2
        col = index % 2
        x = MARGIN_X + 8 * mm + col * 83 * mm
        yy = row_top - row * 20 * mm
        c.setFillColor(TEAL)
        c.circle(x + 2 * mm, yy - 2 * mm, 1.6 * mm, fill=1, stroke=0)
        draw_paragraph(c, title, x + 7 * mm, yy + 2 * mm, 28 * mm, CARD_TITLE)
        draw_paragraph(c, text, x + 36 * mm, yy + 2 * mm, 45 * mm, BODY_SMALL)

    y3 = y2 - 58 * mm
    draw_paragraph(c, "本周交付", MARGIN_X, y3 + 12 * mm, 40 * mm, SUBHEAD)
    deliverables = [
        "四阶段漏斗 CLI 与可恢复状态管理",
        "未来扰动、Newey-West、市值中性化、可交易性和滚动风险模块串联",
        "83 条 Alpha101 全量运行及 79 条组合层面产物",
        "跨阶段 funnel_summary.csv、逐阶段 metrics/details/plots/code",
    ]
    for index, item in enumerate(deliverables):
        x = MARGIN_X + (index % 2) * 88 * mm
        yy = y3 - 8 * mm - (index // 2) * 15 * mm
        c.setFillColor(NAVY)
        c.roundRect(x, yy, 6 * mm, 6 * mm, 1.5 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 7)
        c.setFillColor(WHITE)
        c.drawCentredString(x + 3 * mm, yy + 1.8 * mm, str(index + 1))
        draw_paragraph(c, item, x + 9 * mm, yy + 7 * mm, 75 * mm, BODY_SMALL)
    c.showPage()


def architecture_page(c: canvas.Canvas) -> None:
    page_header(c, 3, "模块化架构", "02 / SYSTEM DESIGN")
    top = section_title(
        c,
        "02",
        "模块化设计",
        "表达式执行、评价方法、流程编排、结果持久化相互解耦，新增方法不需要修改主流程指标 schema。",
    )

    layers = [
        ("数据契约层", "宽矩阵 OHLCV / amount / VWAP proxy / market cap；以 close 轴统一对齐", "#E8F1F8"),
        ("表达式引擎", "AST 白名单校验、按需加载数据、统一算子命名空间、因子矩阵计算", "#E7F7F5"),
        ("Evaluator 注册层", "EvaluationState + @evaluation_method；依赖解析；metrics/details/artifacts 合并", "#FFF6DD"),
        ("漏斗编排层", "stage gate、淘汰即停、断点复用、失败隔离、阶段目录独立", "#FBEDEA"),
        ("归档与展示层", "metrics/history、日度明细、图表、可重跑代码、funnel summary", "#EEF0F8"),
    ]
    layer_h = 27 * mm
    y = top - layer_h
    for index, (title, text, fill) in enumerate(layers):
        x_offset = index * 3 * mm
        width = CONTENT_W - 2 * x_offset
        x = MARGIN_X + x_offset
        rounded_panel(c, x, y, width, 20 * mm, fill=HexColor(fill))
        c.setFillColor(NAVY if index != 2 else AMBER)
        c.roundRect(x + 5 * mm, y + 5 * mm, 24 * mm, 10 * mm, 2 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 8.5)
        c.setFillColor(WHITE if index != 2 else NAVY)
        c.drawCentredString(x + 17 * mm, y + 8.3 * mm, title)
        draw_paragraph(c, text, x + 35 * mm, y + 14 * mm, width - 41 * mm, BODY_SMALL)
        if index < len(layers) - 1:
            c.setStrokeColor(MUTED)
            c.setLineWidth(1)
            c.line(PAGE_W / 2, y - 4 * mm, PAGE_W / 2, y - 7 * mm)
            c.line(PAGE_W / 2, y - 7 * mm, PAGE_W / 2 - 2 * mm, y - 5 * mm)
            c.line(PAGE_W / 2, y - 7 * mm, PAGE_W / 2 + 2 * mm, y - 5 * mm)
        y -= layer_h

    note_y = 28 * mm
    rounded_panel(c, MARGIN_X, note_y, CONTENT_W, 28 * mm, fill=NAVY, stroke=NAVY)
    draw_paragraph(
        c,
        "设计收益",
        MARGIN_X + 7 * mm,
        note_y + 21 * mm,
        30 * mm,
        style("white_subhead", size=11, leading=14, color=WHITE),
    )
    draw_paragraph(
        c,
        "方法可组合、输入按需加载、产物 schema 可扩展；同一表达式可在不同阶段目录独立运行，"
        "且历史结果只有在收益定义与配置完全一致时才会复用。",
        MARGIN_X + 38 * mm,
        note_y + 21 * mm,
        CONTENT_W - 45 * mm,
        WHITE_SMALL,
    )
    c.showPage()


def funnel_page(c: canvas.Canvas) -> None:
    page_header(c, 4, "评价流程", "03 / EVALUATION FUNNEL")
    top = section_title(
        c,
        "03",
        "先正确性，后统计，去混杂，最后确认组合",
        "所有步骤共享 next-open 收益标签；任一门槛淘汰后不再计算后续高成本方法。",
    )

    c.setFillColor(NAVY)
    c.roundRect(MARGIN_X, top - 17 * mm, CONTENT_W, 13 * mm, 3 * mm, fill=1, stroke=0)
    c.setFont("STSong-Light", 9)
    c.setFillColor(WHITE)
    c.drawCentredString(
        PAGE_W / 2,
        top - 12.2 * mm,
        "factor[t] 收盘后计算  ->  open[t+1] 入场  ->  open[t+1+horizon] 出场",
    )

    stages = [
        (
            "1",
            "未来函数检查",
            "future_data_perturbation",
            "4 个 checkpoint；扰动 t0 之后的数据，factor[t0] 不得变化",
            "83 -> 83",
            TEAL,
        ),
        (
            "2A",
            "原始 Rank IC",
            "rank_ic + ICIR + Newey-West",
            "日度截面 Spearman IC；5% 双侧显著性，不显著即淘汰",
            "83 -> 80",
            BLUE,
        ),
        (
            "2B",
            "市值中性 IC",
            "OLS residual + Rank IC + Newey-West",
            "每日 factor = a + b*log(cap) + residual；残差复测",
            "80 -> 79",
            AMBER,
        ),
        (
            "3",
            "组合层面确认",
            "tradability + Q1-Q10 + rolling risk",
            "开盘涨跌停/ST 过滤、顶部两组、换手、20/60/252 日风险序列",
            "79 完成",
            RED,
        ),
    ]
    start_y = top - 56 * mm
    box_h = 37 * mm
    for index, (number, title, methods, text, count, accent) in enumerate(stages):
        y = start_y - index * 45 * mm
        rounded_panel(c, MARGIN_X + 15 * mm, y, CONTENT_W - 15 * mm, box_h)
        c.setFillColor(accent)
        c.circle(MARGIN_X + 9 * mm, y + box_h / 2, 7 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 9)
        c.setFillColor(WHITE if accent != AMBER else NAVY)
        c.drawCentredString(MARGIN_X + 9 * mm, y + box_h / 2 - 3, number)
        draw_paragraph(c, title, MARGIN_X + 23 * mm, y + 29 * mm, 44 * mm, SUBHEAD)
        draw_paragraph(c, methods, MARGIN_X + 23 * mm, y + 19 * mm, 70 * mm, CAPTION)
        draw_paragraph(c, text, MARGIN_X + 75 * mm, y + 29 * mm, 82 * mm, BODY_SMALL)
        c.setFillColor(accent)
        c.roundRect(PAGE_W - MARGIN_X - 28 * mm, y + 7 * mm, 22 * mm, 9 * mm, 2 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 8)
        c.setFillColor(WHITE if accent != AMBER else NAVY)
        c.drawCentredString(PAGE_W - MARGIN_X - 17 * mm, y + 10.2 * mm, count)
        if index < len(stages) - 1:
            c.setStrokeColor(LINE)
            c.setLineWidth(2)
            c.line(MARGIN_X + 9 * mm, y - 1 * mm, MARGIN_X + 9 * mm, y - 8 * mm)

    c.setFillColor(PANEL)
    c.roundRect(MARGIN_X, 18 * mm, CONTENT_W, 23 * mm, 3 * mm, fill=1, stroke=0)
    draw_paragraph(
        c,
        "统一标签：RETURN_DEFINITION = open[t+1+horizon] / open[t+1] - 1；"
        "默认 horizon=1，即 open[t+2] / open[t+1] - 1。",
        MARGIN_X + 7 * mm,
        34 * mm,
        CONTENT_W - 14 * mm,
        BODY_SMALL,
    )
    c.showPage()


def principles_page(c: canvas.Canvas) -> None:
    page_header(c, 5, "方法原理", "04 / METHOD PRINCIPLES")
    top = section_title(
        c,
        "04",
        "每个评价环节解决什么问题",
        "正确性、预测力、独立性和可交易性分别回答不同问题，不能用单一指标替代。",
    )

    cards = [
        (
            "未来数据扰动",
            "问题：表达式是否偷看 t 之后的信息？",
            "做法：固定 t0，仅随机改变未来有限值并重算。若 factor[t0] 改变，说明计算链存在前视依赖。",
            "门槛：4/4 checkpoint 全部不变",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "Rank IC + Newey-West",
            "问题：因子与未来收益是否存在稳定截面相关？",
            "做法：逐日计算 Spearman IC；ICIR 衡量均值相对波动；HAC 标准误检验均值是否显著异于 0。",
            "门槛：双侧 p < 0.05",
            BLUE,
            HexColor("#E8F1F8"),
        ),
        (
            "市值中性化复测",
            "问题：预测力是否只是市值风格暴露？",
            "做法：每日截面 OLS，取 factor 对 log(cap) 回归后的 residual，再重复 IC 显著性检验。",
            "门槛：中性 IC 双侧 p < 0.05",
            AMBER,
            AMBER_LIGHT,
        ),
        (
            "Quantile 与滚动风险",
            "问题：信号能否形成可执行的 long-only 组合？",
            "做法：入场开盘涨跌停/ST 过滤，Q1-Q10 等权；关注 Q9+Q10 收益、换手和 20/60/252 日 Sharpe/回撤分布。",
            "门槛：v1 人工审查，尚未自动淘汰",
            RED,
            RED_LIGHT,
        ),
    ]
    card_w = (CONTENT_W - 6 * mm) / 2
    card_h = 78 * mm
    for index, (title, question, method, gate, accent, fill) in enumerate(cards):
        row = index // 2
        col = index % 2
        x = MARGIN_X + col * (card_w + 6 * mm)
        y = top - card_h - row * (card_h + 7 * mm)
        rounded_panel(c, x, y, card_w, card_h, fill=fill, stroke=accent)
        c.setFillColor(accent)
        c.roundRect(x, y + card_h - 12 * mm, card_w, 12 * mm, 3 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 11)
        c.setFillColor(WHITE if accent != AMBER else NAVY)
        c.drawString(x + 6 * mm, y + card_h - 8 * mm, title)
        draw_paragraph(c, question, x + 6 * mm, y + card_h - 18 * mm, card_w - 12 * mm, CARD_TITLE)
        draw_paragraph(c, method, x + 6 * mm, y + card_h - 33 * mm, card_w - 12 * mm, BODY_SMALL)
        c.setStrokeColor(accent)
        c.line(x + 6 * mm, y + 17 * mm, x + card_w - 6 * mm, y + 17 * mm)
        draw_paragraph(
            c,
            gate,
            x + 6 * mm,
            y + 13 * mm,
            card_w - 12 * mm,
            style("gate", size=8, leading=11, color=accent),
        )
    c.showPage()


def results_page(c: canvas.Canvas, data: dict[str, object]) -> None:
    eliminated = data["eliminated"]
    assert isinstance(eliminated, pd.DataFrame)
    page_header(c, 6, "本次运行效果", "05 / RUN RESULTS")
    top = section_title(
        c,
        "05",
        "统计筛选成功识别 4 条弱或暴露驱动因子",
        "83 条均通过未来函数检查；原始 IC 淘汰 3 条，市值中性 IC 再淘汰 1 条，79 条完成组合评价。",
    )

    funnel_counts = [
        ("未来检查", 83, TEAL),
        ("原始 IC", 80, BLUE),
        ("市值中性", 79, AMBER),
        ("组合完成", 79, RED),
    ]
    x = MARGIN_X
    max_width = 78 * mm
    base_y = top - 13 * mm
    for index, (label, count, accent) in enumerate(funnel_counts):
        width = max_width * count / 83
        y = base_y - index * 14 * mm
        c.setFillColor(PANEL)
        c.roundRect(x, y - 8 * mm, max_width, 9 * mm, 2 * mm, fill=1, stroke=0)
        c.setFillColor(accent)
        c.roundRect(x, y - 8 * mm, width, 9 * mm, 2 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 8)
        c.setFillColor(INK)
        c.drawString(x + max_width + 5 * mm, y - 5 * mm, f"{label}  {count}")

    right_x = MARGIN_X + 105 * mm
    metric_card(c, right_x, top - 34 * mm, 32 * mm, "4", "统计门槛淘汰", accent=RED)
    metric_card(c, right_x + 37 * mm, top - 34 * mm, 32 * mm, "0", "运行失败", accent=TEAL)
    metric_card(c, right_x, top - 63 * mm, 32 * mm, "79", "组合评价完成", accent=BLUE)
    metric_card(c, right_x + 37 * mm, top - 63 * mm, 32 * mm, "12", "顶部两组领先", accent=AMBER)

    table_top = top - 84 * mm
    draw_paragraph(c, "淘汰因子明细", MARGIN_X, table_top, 60 * mm, SUBHEAD)
    rows = [["因子", "淘汰位置", "IC mean", "ICIR", "原始 p", "中性 IC", "中性 p"]]
    for row in eliminated.itertuples(index=False):
        rows.append(
            [
                row.factor_name,
                "原始 IC" if row.terminal_stage == "stage2_ic" else "市值中性",
                f"{row.raw_ic_mean:.4f}",
                f"{row.raw_ir:.3f}",
                f"{row.raw_nw_p_value:.4g}",
                "-" if pd.isna(row.neutral_ic_mean) else f"{row.neutral_ic_mean:.4f}",
                "-" if pd.isna(row.neutral_nw_p_value) else f"{row.neutral_nw_p_value:.4g}",
            ]
        )
    table = Table(
        rows,
        colWidths=[34 * mm, 24 * mm, 21 * mm, 18 * mm, 23 * mm, 22 * mm, 23 * mm],
        rowHeights=[9 * mm] + [10 * mm] * (len(rows) - 1),
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("BACKGROUND", (0, 1), (-1, -1), WHITE),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    _, table_h = table.wrap(CONTENT_W, 60 * mm)
    table.drawOn(c, MARGIN_X, table_top - 8 * mm - table_h)

    callout_y = table_top - 8 * mm - table_h - 38 * mm
    rounded_panel(c, MARGIN_X, callout_y, CONTENT_W, 30 * mm, fill=AMBER_LIGHT, stroke=AMBER)
    draw_paragraph(c, "典型案例：Alpha007", MARGIN_X + 7 * mm, callout_y + 23 * mm, 48 * mm, CARD_TITLE)
    draw_paragraph(
        c,
        "原始 IC=-0.0068、ICIR=-0.295、p=6.42e-35，统计上高度显著；"
        "市值中性后 IC=-0.0023、ICIR=-0.015、p=0.4038，显著性完全消失。"
        "这说明原始信号主要由市值暴露驱动，中性化环节发挥了有效去混杂作用。",
        MARGIN_X + 55 * mm,
        callout_y + 23 * mm,
        CONTENT_W - 62 * mm,
        BODY_SMALL,
    )
    c.showPage()


def issues_page(c: canvas.Canvas, data: dict[str, object]) -> None:
    timings = data["timings"]
    assert isinstance(timings, pd.DataFrame)
    stage1_minutes = float(timings.loc["stage1_validity", "total_minutes"])
    total_minutes = float(timings["total_minutes"].sum())
    stage1_share = stage1_minutes / total_minutes
    slowest = str(timings.loc["stage1_validity", "slowest_factor"])
    slowest_seconds = float(timings.loc["stage1_validity", "slowest_seconds"])

    page_header(c, 7, "版本 1.0 复盘", "06 / ISSUES & LEARNINGS")
    top = section_title(
        c,
        "06",
        "1.0 版本问题与本周发现",
        "当前流程已能稳定运行，但计算效率、统计稳健性和 A 股 long-only 表达仍需进一步优化。",
    )

    issues = [
        (
            "01",
            "未来函数检查计算成本高",
            f"阶段 1 最终有效运行累计 {stage1_minutes:.1f} 分钟，占四阶段总计算时间约 {stage1_share:.0%}。"
            f"最慢因子 {slowest} 单条约 {slowest_seconds / 60:.1f} 分钟。原因是每条因子需完整重算 4 次，"
            "rolling apply 类算子进一步放大成本。",
            "改进方向：缓存公共中间结果、优化 ts_argmax/decay_linear、设计增量或截断式扰动。",
            TEAL,
        ),
        (
            "02",
            "Newey-West 参数 L 选择失效",
            "当前使用固定量纲的自协方差阈值 0.05。实跑中原始 IC 的 83/83、"
            "中性 IC 的 80/80 均选择 L=0，HAC 实际退化为普通均值 t 检验，"
            "无法充分修正日度 IC 的序列相关。",
            "改进方向：采用自相关显著带、Andrews/Newey-West plug-in 带宽，并做敏感性对比。",
            BLUE,
        ),
        (
            "03",
            "Quantile 输出需要突出 Q9/Q10",
            "现有输出仍保留 Long-Short 作为通用研究指标，但 A 股无法直接做空且存在融券约束。"
            "本次 79 条存活因子中仅 12 条满足“Q9+Q10 高于所有低分组”，"
            "说明仅看多空端点会高估可执行性。",
            "改进方向：将 Q10、Q9+Q10 的收益、超额、换手和回撤作为 long-only 主视图。",
            AMBER,
        ),
    ]
    card_h = 57 * mm
    for index, (number, title, body, action, accent) in enumerate(issues):
        y = top - card_h - index * (card_h + 7 * mm)
        rounded_panel(c, MARGIN_X, y, CONTENT_W, card_h, fill=WHITE, stroke=accent)
        c.setFillColor(accent)
        c.roundRect(MARGIN_X, y, 24 * mm, card_h, 4 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 18)
        c.setFillColor(WHITE if accent != AMBER else NAVY)
        c.drawCentredString(MARGIN_X + 12 * mm, y + 33 * mm, number)
        draw_paragraph(c, title, MARGIN_X + 31 * mm, y + 48 * mm, 70 * mm, SUBHEAD)
        draw_paragraph(c, body, MARGIN_X + 31 * mm, y + 36 * mm, CONTENT_W - 39 * mm, BODY_SMALL)
        c.setFillColor(PANEL)
        c.roundRect(
            MARGIN_X + 31 * mm,
            y + 7 * mm,
            CONTENT_W - 39 * mm,
            13 * mm,
            2 * mm,
            fill=1,
            stroke=0,
        )
        draw_paragraph(
            c,
            action,
            MARGIN_X + 35 * mm,
            y + 17 * mm,
            CONTENT_W - 47 * mm,
            style("action", size=7.8, leading=10.5, color=accent),
        )
    c.showPage()


def next_week_page(c: canvas.Canvas) -> None:
    page_header(c, 8, "下周计划", "07 / NEXT WEEK")
    top = section_title(
        c,
        "07",
        "从可运行 1.0 走向可决策 1.1",
        "下周重点不是继续堆指标，而是修正关键统计口径、建立综合评价函数并完成统一可视化入口。",
    )

    workstreams = [
        (
            "P0",
            "修复已发现问题",
            [
                "重构 Newey-West 带宽 L 的选择，并回归验证显著性结果",
                "优化未来扰动测试与 rolling apply 性能",
                "按 IC 符号统一 long direction，突出 Q9/Q10 long-only 结果",
            ],
            "验收：同一因子可输出 L 敏感性、阶段耗时下降、正负 IC 方向一致。",
            RED,
        ),
        (
            "P1",
            "增加 Fitness 评价函数",
            [
                "组合 IC 幅度、ICIR、显著性、组间排序、换手和滚动风险",
                "区分 hard gate 与 soft score，避免单一 IR 排名",
                "预留成本、holdout、相关性和增量价值接口",
            ],
            "验收：形成可解释、可配置、可批量排序的 Fitness v0.1。",
            BLUE,
        ),
        (
            "P1",
            "建设可视化因子评价平台",
            [
                "统一展示漏斗状态、IC、中性 IC、Q9/Q10 与滚动风险",
                "支持因子横向比较、阶段筛选与代理口径标识",
                "直接读取标准输出目录，不复制计算逻辑",
            ],
            "验收：一个入口完成因子筛选、明细下钻和图表审查。",
            TEAL,
        ),
    ]
    card_h = 58 * mm
    for index, (priority, title, bullets, acceptance, accent) in enumerate(workstreams):
        y = top - card_h - index * (card_h + 8 * mm)
        rounded_panel(c, MARGIN_X, y, CONTENT_W, card_h, fill=WHITE, stroke=LINE)
        c.setFillColor(accent)
        c.roundRect(MARGIN_X + 6 * mm, y + card_h - 14 * mm, 17 * mm, 7 * mm, 2 * mm, fill=1, stroke=0)
        c.setFont("STSong-Light", 8.5)
        c.setFillColor(WHITE)
        c.drawCentredString(MARGIN_X + 14.5 * mm, y + card_h - 11.5 * mm, priority)
        draw_paragraph(c, title, MARGIN_X + 29 * mm, y + card_h - 6 * mm, 80 * mm, SUBHEAD)
        for bullet_index, bullet in enumerate(bullets):
            yy = y + card_h - 23 * mm - bullet_index * 12 * mm
            c.setFillColor(accent)
            c.circle(MARGIN_X + 32 * mm, yy + 1 * mm, 1.2 * mm, fill=1, stroke=0)
            draw_paragraph(c, bullet, MARGIN_X + 36 * mm, yy + 5 * mm, CONTENT_W - 47 * mm, BODY_SMALL)
        c.setFillColor(PANEL)
        c.roundRect(
            MARGIN_X + 111 * mm,
            y + 8 * mm,
            CONTENT_W - 119 * mm,
            35 * mm,
            2 * mm,
            fill=1,
            stroke=0,
        )
        draw_paragraph(
            c,
            "验收标准",
            MARGIN_X + 117 * mm,
            y + 37 * mm,
            38 * mm,
            style("accept_head", size=8.5, leading=11, color=accent),
        )
        draw_paragraph(
            c,
            acceptance,
            MARGIN_X + 117 * mm,
            y + 27 * mm,
            CONTENT_W - 131 * mm,
            CAPTION,
        )

    c.setFillColor(NAVY)
    c.roundRect(MARGIN_X, 18 * mm, CONTENT_W, 23 * mm, 3 * mm, fill=1, stroke=0)
    draw_paragraph(
        c,
        "目标状态：让评价体系从“能跑完”升级为“结果稳健、方向可交易、结论可解释、审查可视化”。",
        MARGIN_X + 8 * mm,
        34 * mm,
        CONTENT_W - 16 * mm,
        style("closing", size=9.5, leading=13, color=WHITE, align=TA_CENTER),
    )
    c.showPage()


def build_report(results_dir: Path, output_path: Path) -> Path:
    register_fonts()
    data = load_report_data(results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle("因子评价体系搭建工作周报")
    c.setAuthor("Factor Mining Project")
    c.setSubject("模块化因子评价流程与 Alpha101 全量运行复盘")
    cover_page(c, data)
    overview_page(c, data)
    architecture_page(c)
    funnel_page(c)
    principles_page(c)
    results_page(c, data)
    issues_page(c, data)
    next_week_page(c)
    c.save()
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    required = (
        results_dir / "funnel_summary.csv",
        results_dir / "funnel_status.csv",
        results_dir / "stage2_ic" / "metrics.csv",
        results_dir / "stage2b_neutral" / "metrics.csv",
        results_dir / "stage3_portfolio" / "metrics.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")
    path = build_report(results_dir, output_path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
