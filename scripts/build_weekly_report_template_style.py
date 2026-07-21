#!/usr/bin/env python3
"""Build a plain 16:9 weekly report matching 报告模版.pdf."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
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
    / "outputs"
    / "pdf"
    / "factor_evaluation_weekly_report_2026-07-03.pdf"
)

PAGE_W = 960.0
PAGE_H = 540.0
LEFT = 58.0
RIGHT = PAGE_W - 58.0
TOP = 475.0

BLACK = HexColor("#292929")
DARK = HexColor("#444444")
GRAY = HexColor("#777777")
LIGHT = HexColor("#D7D7D7")
PALE = HexColor("#F5F5F5")
RED = HexColor("#E53935")
BLUE = HexColor("#4E79A7")
TEAL = HexColor("#59A14F")


def register_font() -> None:
    pdfmetrics.registerFont(
        TTFont(
            "ReportSans",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        )
    )


def pstyle(
    name: str,
    size: float,
    *,
    leading: float | None = None,
    color: colors.Color = DARK,
    align: int = TA_LEFT,
) -> ParagraphStyle:
    return ParagraphStyle(
        name,
        fontName="ReportSans",
        fontSize=size,
        leading=leading or size * 1.45,
        textColor=color,
        alignment=align,
        wordWrap="CJK",
    )


TITLE = pstyle("title", 28, leading=36, color=BLACK)
SUBTITLE = pstyle("subtitle", 14, leading=20, color=RED)
BODY = pstyle("body", 15, leading=25)
BODY_SMALL = pstyle("body_small", 12.5, leading=20)
NOTE = pstyle("note", 11, leading=17, color=GRAY)


def draw_paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    style: ParagraphStyle = BODY,
    max_height: float = 400,
) -> float:
    paragraph = Paragraph(text, style)
    _, height = paragraph.wrap(width, max_height)
    paragraph.drawOn(c, x, top - height)
    return height


def slide_title(
    c: canvas.Canvas,
    title: str,
    subtitle: str | None = None,
) -> float:
    draw_paragraph(c, title, LEFT, TOP, 780, TITLE)
    if subtitle:
        draw_paragraph(c, subtitle, LEFT, TOP - 42, 820, NOTE)
        return TOP - 76
    return TOP - 58


def bullet(
    c: canvas.Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    *,
    level: int = 0,
    color: colors.Color = DARK,
    font_size: float = 14,
) -> float:
    indent = level * 24
    radius = 3.0 if level == 0 else 2.0
    c.setFillColor(color)
    c.circle(x + indent + 4, top - 8, radius, fill=1, stroke=0)
    style = pstyle(
        f"bullet_{level}_{font_size}",
        font_size,
        leading=font_size * 1.55,
        color=color,
    )
    height = draw_paragraph(
        c,
        text,
        x + indent + 16,
        top,
        width - indent - 16,
        style,
    )
    return max(height, font_size * 1.55)


def footer(c: canvas.Canvas, page_number: int) -> None:
    c.setFont("ReportSans", 8)
    c.setFillColor(HexColor("#A0A0A0"))
    c.drawRightString(RIGHT, 19, str(page_number))


def finish_slide(c: canvas.Canvas, page_number: int) -> None:
    footer(c, page_number)
    c.showPage()


def load_data(results_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(results_dir / "funnel_summary.csv")
    stage2 = pd.read_csv(results_dir / "stage2_ic" / "metrics.csv")
    stage2b = pd.read_csv(results_dir / "stage2b_neutral" / "metrics.csv")
    stage3 = pd.read_csv(results_dir / "stage3_portfolio" / "metrics.csv")
    status = pd.read_csv(results_dir / "funnel_status.csv")
    evaluated = status.loc[status["execution_state"].eq("evaluated")].copy()
    latest = evaluated.groupby(["factor_name", "stage"], sort=False).tail(1)
    timing: dict[str, dict[str, object]] = {}
    for stage_name, rows in latest.groupby("stage", sort=False):
        slowest = rows.loc[rows["elapsed_seconds"].idxmax()]
        timing[stage_name] = {
            "total_minutes": float(rows["elapsed_seconds"].sum() / 60),
            "median_seconds": float(rows["elapsed_seconds"].median()),
            "slowest_factor": str(slowest["factor_name"]),
            "slowest_seconds": float(slowest["elapsed_seconds"]),
        }
    eliminated = summary.loc[summary["final_status"].eq("eliminated")].copy()
    return {
        "summary": summary,
        "stage2": stage2,
        "stage2b": stage2b,
        "stage3": stage3,
        "timing": timing,
        "eliminated": eliminated,
        "results_dir": results_dir,
    }


def cover(c: canvas.Canvas) -> None:
    draw_paragraph(c, "因子评价体系搭建", 180, 330, 620, pstyle("cover", 34, leading=44, color=BLACK))
    draw_paragraph(
        c,
        "06.29 - 07.03 工作学习汇报",
        180,
        275,
        560,
        pstyle("cover_sub", 15, leading=22, color=RED),
    )
    c.setFont("ReportSans", 11)
    c.setFillColor(DARK)
    c.drawString(180, 145, "作者：黄钜原")
    c.showPage()


def goals(c: canvas.Canvas, page: int) -> None:
    y = slide_title(c, "项目目标 & 范围")
    bullets = [
        "建立统一、可复用的单因子评价流程，而不是为每个因子单独写回测脚本",
        "保证所有环节使用同一收益标签：open[t+1+horizon] / open[t+1] - 1",
        "先验证正确性，再做统计筛选，再去除市值混杂，最后做组合层面确认",
        "当前样本：2013-01-14 至 2026-06-26；A 股宽矩阵数据；默认 horizon=1",
        "本次因子集：83 条当前可运行 Alpha101，其中 30 条使用 VWAP proxy",
        "每个阶段独立输出目录，结果可审计、可恢复、可重复运行",
    ]
    for text in bullets:
        used = bullet(c, text, LEFT, y, 830)
        y -= used + 14
    finish_slide(c, page)


def architecture(c: canvas.Canvas, page: int) -> None:
    y = slide_title(c, "模块化设计")
    layers = [
        ("数据契约层", "OHLCV、amount、VWAP proxy、market cap；以 close 矩阵统一对齐"),
        ("表达式引擎", "AST 白名单校验、按需加载矩阵、公共算子命名空间、因子矩阵计算"),
        ("评价方法层", "@evaluation_method + EvaluationState；方法独立贡献 metrics/details/artifacts"),
        ("漏斗编排层", "依赖解析、阶段 gate、淘汰即停、失败隔离、断点复用"),
        ("结果归档层", "metrics/history、日度明细、图表、可重跑代码和 funnel_summary"),
    ]
    box_x = 56
    box_w = 425
    box_h = 52
    for index, (name, text) in enumerate(layers):
        yy = y - index * 64 - box_h
        c.setStrokeColor(HexColor("#AFAFAF"))
        c.setLineWidth(1)
        c.rect(box_x, yy, box_w, box_h, fill=0, stroke=1)
        c.setFont("ReportSans", 12.5)
        c.setFillColor(BLACK)
        c.drawString(box_x + 14, yy + 31, name)
        draw_paragraph(
            c,
            text,
            box_x + 120,
            yy + 39,
            box_w - 134,
            pstyle(f"architecture_{index}", 10.5, leading=14, color=GRAY),
        )
        if index < len(layers) - 1:
            c.setStrokeColor(GRAY)
            arrow_x = box_x + box_w / 2
            c.line(arrow_x, yy - 3, arrow_x, yy - 12)
            c.line(arrow_x, yy - 12, arrow_x - 4, yy - 8)
            c.line(arrow_x, yy - 12, arrow_x + 4, yy - 8)

    directory_x = 520
    c.setFont("ReportSans", 14)
    c.setFillColor(BLACK)
    c.drawString(directory_x, y - 6, "目前项目目录")
    draw_paragraph(
        c,
        "/Users/huangjuyuan/Desktop/因子挖掘:评价",
        directory_x,
        y - 30,
        380,
        pstyle("project_root", 9.5, leading=13, color=GRAY),
    )
    directories = [
        (0, "data/", "市场宽矩阵与数据清单"),
        (0, "src/", "核心代码"),
        (1, "evaluators/", "评价方法与依赖注册"),
        (1, "engine.py", "表达式计算与流程执行"),
        (1, "returns.py", "统一收益标签"),
        (1, "reporting/", "结果展示"),
        (0, "scripts/", "运行与批处理入口"),
        (0, "tests/", "合约及方法测试"),
        (0, "docs/", "项目说明文档"),
        (0, "factor_registry/", "统一因子定义来源"),
        (0, "outputs/", "评价结果与报告"),
    ]
    directory_y = y - 62
    for level, path, description in directories:
        xx = directory_x + level * 19
        c.setFont("ReportSans", 10.8)
        c.setFillColor(DARK)
        c.drawString(xx, directory_y, path)
        path_width = c.stringWidth(path, "ReportSans", 10.8)
        c.setFont("ReportSans", 9.8)
        c.setFillColor(GRAY)
        c.drawString(xx + path_width + 8, directory_y, description)
        directory_y -= 24
    finish_slide(c, page)


def funnel(c: canvas.Canvas, page: int) -> None:
    y = slide_title(
        c,
        "因子评价流程总览",
        "因子在 t 日收盘后计算，t+1 开盘入场；任何阶段淘汰后不再运行后续步骤。",
    )
    stages = [
        ("环节 1", "未来函数检查", "83 -> 83"),
        ("环节 2A", "原始 Rank IC 显著性", "83 -> 80"),
        ("环节 2B", "市值中性 IC 复测", "80 -> 79"),
        ("环节 3", "Quantile 与组合风险确认", "79 完成"),
    ]
    x = 48
    gap = 18
    width = (PAGE_W - 96 - gap * 3) / 4
    for index, (number, name, count) in enumerate(stages):
        xx = x + index * (width + gap)
        c.setStrokeColor(HexColor("#A9A9A9"))
        c.setFillColor(colors.white)
        c.rect(xx, 230, width, 125, fill=1, stroke=1)
        c.setFont("ReportSans", 12)
        c.setFillColor(RED if index < 3 else DARK)
        c.drawString(xx + 14, 325, number)
        draw_paragraph(c, name, xx + 14, 300, width - 28, pstyle(f"stage_{index}", 15, leading=22, color=BLACK))
        c.setFont("ReportSans", 13)
        c.setFillColor(GRAY)
        c.drawString(xx + 14, 250, count)
        if index < len(stages) - 1:
            arrow_x = xx + width + 4
            c.setStrokeColor(GRAY)
            c.line(arrow_x, 292, arrow_x + gap - 8, 292)
            c.line(arrow_x + gap - 8, 292, arrow_x + gap - 14, 297)
            c.line(arrow_x + gap - 8, 292, arrow_x + gap - 14, 287)
    draw_paragraph(
        c,
        "RETURN_DEFINITION = open[t+1+horizon] / open[t+1] - 1；默认 horizon=1，即 open[t+2] / open[t+1] - 1。",
        90,
        165,
        780,
        pstyle("return_definition", 13.5, leading=22, color=DARK),
    )
    finish_slide(c, page)


def stage1(c: canvas.Canvas, page: int, data: dict[str, object]) -> None:
    timing = data["timing"]
    assert isinstance(timing, dict)
    row = timing["stage1_validity"]
    y = slide_title(c, "环节 1：未来函数检查")
    paragraphs = [
        "目的：检查因子表达式是否使用 t 日之后的数据。若存在前视依赖，后面的 IC 和回测结果都会失真。",
        "方法：选取 4 个 checkpoint t0；保留 t <= t0 的输入不变，对 t > t0 的有限值做确定性随机扰动，再比较 factor[t0]。",
        "门槛：任意 checkpoint 的因子值发生变化即淘汰；4 个 checkpoint 全部不变才通过。",
        "结果：修复非有限值清洗不一致造成的假阳性后，83 条因子全部通过。",
    ]
    for text in paragraphs:
        used = bullet(c, text, LEFT, y, 835)
        y -= used + 18
    draw_paragraph(
        c,
        f"成本观察：最终有效运行累计约 {row['total_minutes']:.1f} 分钟；"
        f"最慢因子 {row['slowest_factor']} 单条约 {float(row['slowest_seconds']) / 60:.1f} 分钟。",
        LEFT + 16,
        115,
        815,
        pstyle("stage1_cost", 13, leading=22, color=RED),
    )
    finish_slide(c, page)


def rank_ic(c: canvas.Canvas, page: int, data: dict[str, object]) -> None:
    stage2 = data["stage2"]
    assert isinstance(stage2, pd.DataFrame)
    y = slide_title(c, "环节 2A：Rank IC 显著性筛选")
    items = [
        "逐日计算因子值与未来收益之间的截面 Spearman Rank IC。",
        "IC mean 衡量平均方向与幅度；ICIR = IC mean / IC std，衡量稳定性。",
        "使用 Newey-West HAC 标准误检验 IC 均值是否显著异于 0。",
        "当前 v1 门槛：双侧 p < 0.05；不显著即淘汰。",
    ]
    for text in items:
        used = bullet(c, text, LEFT, y, 835)
        y -= used + 18

    c.setStrokeColor(LIGHT)
    c.line(90, 190, 870, 190)
    draw_paragraph(
        c,
        "本次结果",
        90,
        165,
        130,
        pstyle("rank_result_head", 16, leading=22, color=BLACK),
    )
    draw_paragraph(
        c,
        "83 条进入，80 条通过；Alpha028、Alpha085、Alpha099 因 p >= 0.05 被淘汰。",
        235,
        165,
        620,
        pstyle("rank_result", 14, leading=23, color=RED),
    )
    lag_zero = int(stage2["nw_ic_lag"].eq(0).sum())
    draw_paragraph(
        c,
        f"注意：{lag_zero}/{len(stage2)} 条原始 IC 序列均选择 L=0，说明当前带宽选择规则需要重构。",
        235,
        125,
        620,
        NOTE,
    )
    finish_slide(c, page)


def neutralization(c: canvas.Canvas, page: int) -> None:
    y = slide_title(c, "环节 2B：市值中性化后复测")
    items = [
        "目的：判断因子信息是否只是小市值或大市值风格暴露。",
        "每日截面执行 OLS：factor_i = intercept + beta * log(cap_i) + residual_i。",
        "使用 residual 作为新的工作因子，重复 Rank IC、ICIR 和 Newey-West 检验。",
        "同时保留每日 log(cap) beta 和 R²，用于诊断因子的市值暴露程度。",
        "Alpha007 暴露了一个具体实现问题：当前项目把 adv20 映射为平均成交额，"
        "把 volume 映射为成交量，再直接比较，属于数据口径理解和映射错误。",
        "结果：Alpha007 原始 p=6.42e-35，中性化后 p=0.4038。显著性随市值暴露被剔除，"
        "说明原始信号主要来自错误口径引入的价格/规模成分。",
    ]
    for text in items:
        used = bullet(c, text, LEFT, y, 835)
        y -= used + 13
    draw_paragraph(
        c,
        "这次中性化不仅筛选了因子，也起到了实现诊断作用：它揪出了一个原始 IC 看似显著、"
        "实际由错误数据口径和市值相关暴露支撑的结果。",
        LEFT + 16,
        88,
        820,
        pstyle("neutral_note", 13.5, leading=22, color=RED),
    )
    finish_slide(c, page)


def quantile(c: canvas.Canvas, page: int) -> None:
    y = slide_title(c, "环节 3：Quantile 回测与组合风险")
    items = [
        "可交易性过滤：若入场日 t+1 开盘触及涨跌停代理边界或为 ST，将该标的因子值置为 NaN。",
        "按每日因子值从低到高划分 Q1-Q10，计算各组等权收益和累计净值。",
        "重点检查 Q9+Q10 是否高于所有较低分组，并计算顶部两组单边换手率。",
        "Rolling Sharpe：20/60/252 日窗口，输出 pos_share / min / median。",
        "Rolling Drawdown：20/60/252 日窗口，输出 worst / median；完整序列保存在 details。",
    ]
    for text in items:
        used = bullet(c, text, LEFT, y, 835)
        y -= used + 14
    draw_paragraph(
        c,
        "A 股约束：Long-Short 仅保留为研究诊断。后续应将 Q10 和 Q9+Q10 的 long-only 表现设为主视图。",
        LEFT + 16,
        90,
        820,
        pstyle("quantile_note", 13.5, leading=22, color=RED),
    )
    finish_slide(c, page)


def result_summary(c: canvas.Canvas, page: int, data: dict[str, object]) -> None:
    eliminated = data["eliminated"]
    assert isinstance(eliminated, pd.DataFrame)
    slide_title(c, "统计筛选结果总结")
    rows = [["factor", "淘汰位置", "IC mean", "ICIR", "原始 p", "中性 IC", "中性 p"]]
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
        colWidths=[160, 110, 95, 85, 110, 100, 110],
        rowHeights=[38] + [48] * 4,
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "ReportSans"),
                ("FONTSIZE", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#3F3F3F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.6, HexColor("#B5B5B5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    table.wrapOn(c, 820, 300)
    table.drawOn(c, 65, 170)
    draw_paragraph(
        c,
        "当前结果：3 条因子在原始 IC 阶段统计淘汰；Alpha007 的“淘汰”实为中性化发现的"
        "口径映射问题，修正后需要重新评价。",
        82,
        130,
        800,
        pstyle("result_final", 14, leading=22, color=RED),
    )
    finish_slide(c, page)


def alpha007(c: canvas.Canvas, page: int) -> None:
    slide_title(c, "Alpha007：口径误读与市值中性化诊断")

    c.setFont("ReportSans", 13)
    c.setFillColor(BLACK)
    c.drawString(72, 406, "原论文公式")
    c.setFillColor(PALE)
    c.rect(72, 350, 816, 42, fill=1, stroke=0)
    draw_paragraph(
        c,
        "((adv20 &lt; volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) "
        "* sign(delta(close, 7))) : -1)",
        88,
        380,
        785,
        pstyle("alpha007_paper", 12.5, leading=18, color=DARK),
    )

    c.setFont("ReportSans", 13)
    c.setFillColor(BLACK)
    c.drawString(72, 325, "项目当前直译实现")
    c.setFillColor(PALE)
    c.rect(72, 269, 816, 42, fill=1, stroke=0)
    draw_paragraph(
        c,
        "where(adv(amt, 20) &lt; vol, ((-1 * ts_rank(abs(delta(c, 7)), 60)) "
        "* sign(delta(c, 7))), -1.0)",
        88,
        299,
        785,
        pstyle("alpha007_local", 12.5, leading=18, color=DARK),
    )

    c.setFont("ReportSans", 14)
    c.setFillColor(BLACK)
    c.drawString(72, 236, "问题根源：数据口径理解和映射错误")
    left_notes = [
        "把 adv20 落到 amt、volume 落到 vol 后直接比较，量纲不一致。",
        "近似有 amt = price * vol，因此条件不再是单纯判断“今日是否放量”。",
        "错误比较引入价格尺度，并可能与市值暴露耦合，因子也可能大量退化为 -1。",
    ]
    y = 212
    for text in left_notes:
        used = bullet(c, text, 72, y, 405, font_size=11.5)
        y -= used + 5

    c.setFont("ReportSans", 14)
    c.setFillColor(BLACK)
    c.drawString(510, 236, "中性化如何把问题揪出来")
    right_notes = [
        "原始检验：p=6.42e-35，表面上高度显著。",
        "剔除 log(cap) 暴露后：p=0.4038，显著性消失。",
        "诊断结论：原始信号主要由市值相关成分支撑；这是实现问题的证据，不是对论文因子的否定。",
    ]
    y = 212
    for text in right_notes:
        used = bullet(c, text, 510, y, 378, font_size=11.5)
        y -= used + 5

    draw_paragraph(
        c,
        "修复：分别建立成交量版 ts_mean(vol,20) < vol 与成交额版 adv(amt,20) < amt，"
        "检查条件触发率和 -1 占比后重新跑完整流程。",
        72,
        105,
        816,
        pstyle("alpha007_warning", 12.3, leading=18, color=RED),
    )
    draw_paragraph(
        c,
        "来源：Z. Kakushadze, 101 Formulaic Alphas, Appendix A / A.2 "
        "(https://arxiv.org/pdf/1601.00991)",
        72,
        60,
        816,
        pstyle("alpha007_source", 9.5, leading=14, color=GRAY),
    )
    finish_slide(c, page)


def quantile_example(c: canvas.Canvas, page: int, data: dict[str, object]) -> None:
    results_dir = data["results_dir"]
    assert isinstance(results_dir, Path)
    plot_path = results_dir / "stage3_portfolio" / "plots" / "alpha101_055.png"
    slide_title(c, "Quantile 回测示例：Alpha055")
    if plot_path.is_file():
        image = ImageReader(str(plot_path))
        iw, ih = image.getSize()
        max_w, max_h = 720, 355
        scale = min(max_w / iw, max_h / ih)
        width, height = iw * scale, ih * scale
        c.drawImage(
            image,
            (PAGE_W - width) / 2,
            85,
            width,
            height,
            preserveAspectRatio=True,
            mask="auto",
        )
    draw_paragraph(
        c,
        "Alpha055 满足 top_group_above_every_lower_group；原始 IC=0.0224，ICIR=0.417。"
        "该图仅用于展示现有分组产物，尚未扣除交易成本，也不是最终选因子结论。",
        95,
        72,
        770,
        NOTE,
    )
    finish_slide(c, page)


def issues(c: canvas.Canvas, page: int, data: dict[str, object]) -> None:
    timing = data["timing"]
    stage2 = data["stage2"]
    stage2b = data["stage2b"]
    summary = data["summary"]
    assert isinstance(timing, dict)
    assert isinstance(stage2, pd.DataFrame)
    assert isinstance(stage2b, pd.DataFrame)
    assert isinstance(summary, pd.DataFrame)
    stage1_minutes = float(timing["stage1_validity"]["total_minutes"])
    total_minutes = sum(float(row["total_minutes"]) for row in timing.values())
    top_group_winners = int(
        summary.loc[
            summary["final_status"].eq("completed"),
            "top_group_above_every_lower_group",
        ].eq(True).sum()
    )
    y = slide_title(c, "1.0 版本问题")
    items = [
        (
            "环节 1 计算时间较长",
            f"最终有效运行约 {stage1_minutes:.1f} 分钟，占四阶段总计算时间约 {stage1_minutes / total_minutes:.0%}；"
            "4 次完整重算与 rolling apply 是主要瓶颈。",
        ),
        (
            "Newey-West 参数 L 的选择需要重新设计",
            f"原始 IC {int(stage2.nw_ic_lag.eq(0).sum())}/{len(stage2)}、中性 IC "
            f"{int(stage2b.nw_ic_lag.eq(0).sum())}/{len(stage2b)} 均选择 L=0，HAC 退化为普通 t 检验。",
        ),
        (
            "Quantile 回测需要突出 Q10",
            f"A 股很难直接做空。当前 79 条存活因子中仅 {top_group_winners} 条满足最高组领先，"
            "因此 long-only 表现比 Long-Short 更有实际意义。",
        ),
    ]
    for title, explanation in items:
        used = bullet(c, title, LEFT, y, 835, font_size=15)
        y -= used + 4
        used = bullet(c, explanation, LEFT + 12, y, 810, level=1, color=GRAY, font_size=13)
        y -= used + 28
    finish_slide(c, page)


def next_week(c: canvas.Canvas, page: int) -> None:
    y = slide_title(c, "下周改进计划")
    plans = [
        (
            "1. 修复本周发现的问题",
            [
                "重新设计 Newey-West 带宽 L，并做敏感性对比",
                "优化未来扰动和 rolling apply 性能",
                "为 Alpha007 建立成交量/成交额两个量纲一致版本，并检查条件触发率与 -1 占比",
                "按 IC 方向统一 long direction，强化 Q9/Q10 long-only 输出",
            ],
        ),
        (
            "2. 增加 Fitness 评价函数设计",
            [
                "综合 IC 幅度、ICIR、显著性、组间排序、换手与滚动风险",
                "区分 hard gate 与 soft score，避免只按单一 IR 排名",
            ],
        ),
        (
            "3. 建设一个可视化因子评价平台",
            [
                "统一展示漏斗状态、原始/中性 IC、Q9/Q10、换手和滚动风险",
                "支持因子横向比较与日度明细下钻，直接读取标准输出目录",
            ],
        ),
    ]
    for title, children in plans:
        used = bullet(c, title, LEFT, y, 835, font_size=15)
        y -= used + 5
        for child in children:
            used = bullet(c, child, LEFT + 12, y, 810, level=1, color=GRAY, font_size=12.8)
            y -= used + 2
        y -= 17
    finish_slide(c, page)


def conclusion(c: canvas.Canvas, page: int) -> None:
    y = slide_title(c, "结论")
    items = [
        "完成模块化因子评价体系 1.0，并形成可恢复、可追溯的四阶段评价流程。",
        "83 条 Alpha101 因子完成全量验证，3 条在原始 IC 阶段统计淘汰。",
        "Alpha007 不是普通统计淘汰：成交额与成交量的错误比较引入价格/规模暴露，"
        "市值中性化后显著性消失，从而暴露出实现问题；修正口径后需重跑。",
        "当前主要短板是未来检查成本、Newey-West 带宽和 long-only 组合表达。",
        "下周重点转向统计修正、Fitness 设计与统一可视化平台。",
    ]
    for text in items:
        used = bullet(c, text, LEFT, y, 835)
        y -= used + 24
    finish_slide(c, page)


def build_report(results_dir: Path, output_path: Path) -> Path:
    register_font()
    data = load_data(results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("因子评价体系搭建工作汇报")
    c.setAuthor("黄钜原")
    cover(c)
    goals(c, 2)
    architecture(c, 3)
    funnel(c, 4)
    stage1(c, 5, data)
    rank_ic(c, 6, data)
    neutralization(c, 7)
    quantile(c, 8)
    result_summary(c, 9, data)
    alpha007(c, 10)
    quantile_example(c, 11, data)
    issues(c, 12, data)
    next_week(c, 13)
    conclusion(c, 14)
    c.save()
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    required = [
        results_dir / "funnel_summary.csv",
        results_dir / "funnel_status.csv",
        results_dir / "stage2_ic" / "metrics.csv",
        results_dir / "stage2b_neutral" / "metrics.csv",
        results_dir / "stage3_portfolio" / "metrics.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing report inputs: {missing}")
    print(build_report(results_dir, output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
