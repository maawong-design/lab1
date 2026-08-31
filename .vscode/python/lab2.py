import os
import math
import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Cố định random seed để kết quả có thể tái lập (reproducible)
np.random.seed(42)

# ------------------------------------------------------------------------------
# THƯ MỤC LƯU FILE OUTPUT (tự động, không hard-code theo Linux/Colab)
# - Nếu chạy dưới dạng file .py (Windows/local): lấy đúng thư mục chứa file script.
# - Nếu chạy trong Jupyter/Colab (không có __file__): dùng thư mục làm việc hiện tại.
# ------------------------------------------------------------------------------
try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    OUTPUT_DIR = Path.cwd()

# Tự động tạo thư mục nếu chưa tồn tại (đề phòng trường hợp chỉ định thư mục con)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PHẦN 1: HÀM SINH DỮ LIỆU GIẢ LẬP (SYNTHETIC DATA GENERATION)
# ==============================================================================
def generate_sales_data(n_rows: int = 800) -> pd.DataFrame:
    """
    Sinh ra một DataFrame dữ liệu bán hàng giả lập cho cửa hàng E-commerce.

    Tham số:
        n_rows (int): Số lượng dòng dữ liệu cần sinh (mặc định 800 dòng).

    Trả về:
        pd.DataFrame: Dữ liệu thô chưa qua xử lý.
    """

    # --- 1.1. Mã đơn hàng: DH001, DH002, ... ---
    ma_don_hang = [f"DH{str(i).zfill(3)}" for i in range(1, n_rows + 1)]

    # --- 1.2. Ngày đặt hàng: ngẫu nhiên trong năm 2025 ---
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days
    ngay_dat_hang = [
        start_date + timedelta(days=np.random.randint(0, date_range_days + 1))
        for _ in range(n_rows)
    ]

    # --- 1.3. Danh mục sản phẩm ---
    danh_muc_list = ['Điện tử', 'Thời trang', 'Gia dụng', 'Sách']
    danh_muc = np.random.choice(danh_muc_list, size=n_rows)

    # --- 1.4. Số lượng: 1 đến 10 ---
    so_luong = np.random.randint(1, 11, size=n_rows)

    # --- 1.5. Đơn giá: 50,000 - 5,000,000 VNĐ (làm tròn đến hàng nghìn cho thực tế) ---
    don_gia = np.random.randint(50_000, 5_000_001, size=n_rows)
    don_gia = (don_gia // 1000) * 1000

    # --- 1.6. Trạng thái đơn hàng: Hoàn thành ~80%, còn lại chia đều ---
    trang_thai_list = ['Hoàn thành', 'Đã hủy', 'Đang giao']
    trang_thai_probs = [0.80, 0.10, 0.10]
    trang_thai = np.random.choice(trang_thai_list, size=n_rows, p=trang_thai_probs)

    # --- 1.7. Thành phố ---
    thanh_pho_list = ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ']
    thanh_pho = np.random.choice(thanh_pho_list, size=n_rows)

    # --- Gộp tất cả thành DataFrame ---
    df = pd.DataFrame({
        'Ma_Don_Hang': ma_don_hang,
        'Ngay_Dat_Hang': ngay_dat_hang,
        'DanhMuc': danh_muc,
        'So_Luong': so_luong,
        'Don_Gia': don_gia,
        'Trang_Thai': trang_thai,
        'Thanh_Pho': thanh_pho
    })

    return df


# ==============================================================================
# PHẦN 2: TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
# ==============================================================================
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu và tạo thêm các thuộc tính (feature engineering).

    Tham số:
        df_raw (pd.DataFrame): Dữ liệu thô.

    Trả về:
        pd.DataFrame: Dữ liệu đã xử lý, chỉ gồm đơn hàng KHÔNG bị hủy,
                       có thêm cột Doanh_Thu, Thang, Thang_Nam.
    """

    df = df_raw.copy()

    # 2.1. Đảm bảo cột ngày đúng kiểu datetime
    df['Ngay_Dat_Hang'] = pd.to_datetime(df['Ngay_Dat_Hang'])

    # 2.2. Lọc bỏ các đơn hàng có trạng thái 'Đã hủy'
    #      -> Chỉ giữ lại đơn 'Hoàn thành' và 'Đang giao' để tính doanh thu thực tế
    df_clean = df[df['Trang_Thai'] != 'Đã hủy'].copy()

    # 2.3. Tạo cột Doanh_Thu = Số lượng * Đơn giá
    df_clean['Doanh_Thu'] = df_clean['So_Luong'] * df_clean['Don_Gia']

    # 2.4. Trích xuất thông tin thời gian
    df_clean['Thang'] = df_clean['Ngay_Dat_Hang'].dt.month          # Tháng (1-12)
    df_clean['Thang_Nam'] = df_clean['Ngay_Dat_Hang'].dt.to_period('M').astype(str)  # VD: 2025-01

    # Sắp xếp lại theo ngày đặt hàng cho dễ theo dõi
    df_clean = df_clean.sort_values('Ngay_Dat_Hang').reset_index(drop=True)

    return df_clean


# ==============================================================================
# PHẦN 3: TÍNH TOÁN KPI TỔNG QUAN (EXECUTIVE SUMMARY)
# ==============================================================================
def calculate_kpis(df_clean: pd.DataFrame) -> dict:
    """
    Tính toán các chỉ số kinh doanh cốt lõi (KPI).

    Tham số:
        df_clean (pd.DataFrame): Dữ liệu đã tiền xử lý.

    Trả về:
        dict: Các chỉ số KPI chính.
    """

    # Chỉ tính KPI trên các đơn hàng thực sự "Hoàn thành" (đã giao thành công)
    df_completed = df_clean[df_clean['Trang_Thai'] == 'Hoàn thành']

    tong_doanh_thu = df_completed['Doanh_Thu'].sum()
    tong_don_hang = df_completed['Ma_Don_Hang'].nunique()
    aov = tong_doanh_thu / tong_don_hang if tong_don_hang > 0 else 0

    kpis = {
        'Tổng doanh thu (VNĐ)': tong_doanh_thu,
        'Tổng số đơn hàng thành công': tong_don_hang,
        'Giá trị trung bình / đơn (AOV - VNĐ)': aov
    }

    return kpis


def print_executive_summary(kpis: dict) -> None:
    """In báo cáo tóm tắt KPI ra màn hình theo định dạng dễ đọc."""
    print("=" * 60)
    print("BÁO CÁO TÓM TẮT KINH DOANH (EXECUTIVE SUMMARY) - NĂM 2025")
    print("=" * 60)
    print(f"- Tổng doanh thu toàn kỳ      : {kpis['Tổng doanh thu (VNĐ)']:>18,.0f} VNĐ")
    print(f"- Tổng số đơn hàng thành công : {kpis['Tổng số đơn hàng thành công']:>18,.0f} đơn")
    print(f"- Giá trị TB mỗi đơn (AOV)    : {kpis['Giá trị trung bình / đơn (AOV - VNĐ)']:>18,.0f} VNĐ")
    print("=" * 60 + "\n")


# ==============================================================================
# PHẦN 4: TRỰC QUAN HÓA DỮ LIỆU - DASHBOARD 2x2 (HTML + SVG THUẦN PYTHON)
# ------------------------------------------------------------------------------
# Lý do chuyển từ Matplotlib (PNG) sang HTML/SVG:
#   - Không phụ thuộc việc lưu file ảnh (tránh lỗi FileNotFoundError khi đổi môi trường).
#   - SVG là văn bản (text) nên hiển thị tiếng Việt luôn chuẩn theo font trình duyệt,
#     không lo lỗi thiếu font/ô vuông như khi Matplotlib thiếu font hỗ trợ Unicode.
#   - Có thể mở trực tiếp bằng trình duyệt, phóng to không vỡ nét (vector).
# ==============================================================================

def _dinh_dang_trieu(gia_tri: float) -> str:
    """Định dạng số tiền lớn thành dạng rút gọn 'x,xxx tr' (triệu VNĐ)."""
    return f"{gia_tri / 1e6:,.0f}tr"


def _build_line_chart_svg(series: pd.Series, color: str) -> str:
    """
    Sinh mã SVG cho biểu đồ đường (Xu hướng Doanh thu theo Tháng).
    Sử dụng thẻ <polyline> để nối các điểm dữ liệu.
    """
    width, height = 520, 300
    pad_left, pad_right, pad_top, pad_bottom = 60, 20, 25, 40
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    max_val = series.values.max() if series.values.max() > 0 else 1
    n = len(series)
    step_x = chart_w / (n - 1) if n > 1 else 0

    diem_toa_do = []   # Danh sách toạ độ (x, y) từng điểm
    for i, val in enumerate(series.values):
        x = pad_left + i * step_x
        y = pad_top + chart_h - (val / max_val) * chart_h
        diem_toa_do.append((x, y, val))

    # Đường lưới ngang (grid) + nhãn trục Y
    grid_svg = []
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = pad_top + chart_h - frac * chart_h
        grid_svg.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="#e5e5e5" stroke-dasharray="4,3"/>'
        )
        grid_svg.append(
            f'<text x="{pad_left - 8}" y="{y + 3:.1f}" font-size="9.5" text-anchor="end" fill="#888">'
            f'{_dinh_dang_trieu(max_val * frac)}</text>'
        )

    # Đường nối các điểm (polyline)
    points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in diem_toa_do)
    polyline_svg = f'<polyline points="{points_str}" fill="none" stroke="{color}" stroke-width="2.5"/>'

    # Điểm tròn + nhãn giá trị + nhãn tháng trên trục X
    diem_svg, nhan_svg = [], []
    for i, (x, y, val) in enumerate(diem_toa_do):
        diem_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" stroke="{color}" stroke-width="2.2"/>'
        )
        nhan_svg.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" font-size="9.5" text-anchor="middle" fill="#333">'
            f'{_dinh_dang_trieu(val)}</text>'
        )
        nhan_svg.append(
            f'<text x="{x:.1f}" y="{height - pad_bottom + 18:.1f}" font-size="10.5" '
            f'text-anchor="middle" fill="#555">T{series.index[i]}</text>'
        )

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        {"".join(grid_svg)}
        {polyline_svg}
        {"".join(diem_svg)}
        {"".join(nhan_svg)}
        <line x1="{pad_left}" y1="{pad_top + chart_h:.1f}" x2="{width - pad_right}" y2="{pad_top + chart_h:.1f}" stroke="#999"/>
    </svg>'''


def _build_bar_chart_svg(series: pd.Series, palette: list) -> str:
    """
    Sinh mã SVG cho biểu đồ cột đứng (Doanh thu theo Danh mục).
    Sử dụng thẻ <rect> cho từng cột.
    """
    width, height = 520, 300
    pad_left, pad_right, pad_top, pad_bottom = 55, 20, 30, 45
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    n = len(series)
    max_val = series.values.max() if series.values.max() > 0 else 1
    bar_gap = 22
    bar_width = (chart_w - bar_gap * (n + 1)) / n if n > 0 else 0

    grid_svg = []
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = pad_top + chart_h - frac * chart_h
        grid_svg.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="#e5e5e5" stroke-dasharray="4,3"/>'
        )

    cot_svg, nhan_svg = [], []
    for i, (danh_muc, val) in enumerate(series.items()):
        x = pad_left + bar_gap + i * (bar_width + bar_gap)
        bar_h = (val / max_val) * chart_h
        y = pad_top + chart_h - bar_h
        color = palette[i % len(palette)]
        cot_svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" rx="4"/>'
        )
        nhan_svg.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" font-size="11" '
            f'text-anchor="middle" fill="#333" font-weight="600">{_dinh_dang_trieu(val)}</text>'
        )
        nhan_svg.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - pad_bottom + 18:.1f}" font-size="11" '
            f'text-anchor="middle" fill="#444">{danh_muc}</text>'
        )

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        {"".join(grid_svg)}
        {"".join(cot_svg)}
        {"".join(nhan_svg)}
        <line x1="{pad_left}" y1="{pad_top + chart_h:.1f}" x2="{width - pad_right}" y2="{pad_top + chart_h:.1f}" stroke="#999"/>
    </svg>'''


def _build_donut_chart_svg(series: pd.Series, palette: list) -> tuple:
    """
    Sinh mã SVG cho biểu đồ Donut (Tỷ lệ Doanh thu theo Thành phố).
    Sử dụng thuộc tính stroke-dasharray trên <circle> để tạo từng lát cắt.

    Trả về: (svg_string, legend_html) - chuỗi SVG và khối chú thích (legend) đi kèm.
    """
    size = 260
    r = 80
    cx = cy = size / 2
    stroke_w = 42
    circumference = 2 * math.pi * r
    total = series.values.sum()

    circles_svg = []
    legend_html = []
    cumulative_percent = 0.0

    for i, (thanh_pho, val) in enumerate(series.items()):
        percent = (val / total * 100) if total > 0 else 0
        length = circumference * percent / 100
        dasharray = f"{length:.2f} {circumference - length:.2f}"
        dashoffset = -circumference * cumulative_percent / 100
        color = palette[i % len(palette)]

        circles_svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_w}" stroke-dasharray="{dasharray}" '
            f'stroke-dashoffset="{dashoffset:.2f}" transform="rotate(-90 {cx} {cy})"/>'
        )
        legend_html.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>'
            f'<span>{thanh_pho} — <b>{percent:.1f}%</b></span></div>'
        )
        cumulative_percent += percent

    center_text_svg = (
        f'<text x="{cx}" y="{cy - 6}" font-size="12" text-anchor="middle" fill="#888">Tổng</text>'
        f'<text x="{cx}" y="{cy + 14}" font-size="15" text-anchor="middle" fill="#222" '
        f'font-weight="700">{total / 1e9:,.2f} tỷ</text>'
    )

    svg = f'''<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
        {"".join(circles_svg)}
        {center_text_svg}
    </svg>'''

    return svg, "".join(legend_html)


def _build_hbar_chart_svg(series: pd.Series, palette: list) -> str:
    """
    Sinh mã SVG cho biểu đồ cột ngang (Sản lượng bán theo Danh mục).
    """
    width, height = 520, 300
    pad_left, pad_right, pad_top, pad_bottom = 95, 55, 15, 15
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    n = len(series)
    max_val = series.values.max() if series.values.max() > 0 else 1
    bar_gap = 16
    bar_height = (chart_h - bar_gap * (n + 1)) / n if n > 0 else 0

    cot_svg, nhan_svg = [], []
    for i, (danh_muc, val) in enumerate(series.items()):
        y = pad_top + bar_gap + i * (bar_height + bar_gap)
        bar_w = (val / max_val) * chart_w
        color = palette[i % len(palette)]

        cot_svg.append(
            f'<rect x="{pad_left}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_height:.1f}" '
            f'fill="{color}" rx="4"/>'
        )
        nhan_svg.append(
            f'<text x="{pad_left - 8}" y="{y + bar_height / 2 + 4:.1f}" font-size="11" '
            f'text-anchor="end" fill="#333">{danh_muc}</text>'
        )
        nhan_svg.append(
            f'<text x="{pad_left + bar_w + 8:.1f}" y="{y + bar_height / 2 + 4:.1f}" font-size="11" '
            f'text-anchor="start" fill="#333" font-weight="600">{val:,.0f}</text>'
        )

    return f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        {"".join(cot_svg)}
        {"".join(nhan_svg)}
        <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#999"/>
    </svg>'''


def create_dashboard(df_clean: pd.DataFrame) -> None:
    """
    Tạo Dashboard HTML gồm khối KPI + 4 biểu đồ SVG (bố cục lưới 2x2),
    lưu ra file .html cùng thư mục với script và tự động mở bằng trình duyệt.

    Tham số:
        df_clean (pd.DataFrame): Dữ liệu đã tiền xử lý (đã loại đơn Đã hủy).
    """

    # Palette màu chuyên nghiệp, hài hòa (dùng chung cho mọi biểu đồ)
    color_palette = ['#2E86AB', '#F18F01', '#C73E1D', '#3B8686', '#6A4C93']

    # Chỉ dùng đơn hàng "Hoàn thành" để phân tích doanh thu/hành vi mua thực tế
    df_rev = df_clean[df_clean['Trang_Thai'] == 'Hoàn thành']

    # Tính KPI (tái sử dụng hàm đã có ở Phần 3)
    kpis = calculate_kpis(df_clean)

    # --- Chuẩn bị dữ liệu tổng hợp cho từng biểu đồ ---
    doanh_thu_theo_thang = df_rev.groupby('Thang')['Doanh_Thu'].sum().reindex(range(1, 13), fill_value=0)
    doanh_thu_theo_danhmuc = df_rev.groupby('DanhMuc')['Doanh_Thu'].sum().sort_values(ascending=False)
    doanh_thu_theo_thanhpho = df_rev.groupby('Thanh_Pho')['Doanh_Thu'].sum().sort_values(ascending=False)
    soluong_theo_danhmuc = df_rev.groupby('DanhMuc')['So_Luong'].sum().sort_values(ascending=True)

    # --- Sinh mã SVG cho từng biểu đồ ---
    svg_line = _build_line_chart_svg(doanh_thu_theo_thang, color_palette[0])
    svg_bar = _build_bar_chart_svg(doanh_thu_theo_danhmuc, color_palette)
    svg_donut, legend_donut = _build_donut_chart_svg(doanh_thu_theo_thanhpho, color_palette)
    svg_hbar = _build_hbar_chart_svg(soluong_theo_danhmuc, color_palette)

    # --- Template HTML + CSS (Grid 2x2, font Arial/Helvetica cho tiếng Việt mượt) ---
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Dashboard Bán Hàng 2025</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{
        font-family: Arial, Helvetica, sans-serif;
        background: #f4f6f8;
        margin: 0;
        padding: 30px;
        color: #222;
    }}
    .dashboard-header {{
        text-align: center;
        margin-bottom: 24px;
    }}
    .dashboard-header h1 {{
        font-size: 24px;
        font-weight: 700;
        color: #1b1b1b;
        margin: 0 0 6px 0;
    }}
    .dashboard-header p {{
        color: #777;
        font-size: 13.5px;
        margin: 0;
    }}
    /* --- Khối KPI Executive Summary --- */
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 18px;
        max-width: 1180px;
        margin: 0 auto 28px auto;
    }}
    .kpi-card {{
        background: #fff;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #2E86AB;
    }}
    .kpi-card:nth-child(2) {{ border-left-color: #F18F01; }}
    .kpi-card:nth-child(3) {{ border-left-color: #C73E1D; }}
    .kpi-card .kpi-label {{
        font-size: 12.5px;
        color: #888;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }}
    .kpi-card .kpi-value {{
        font-size: 24px;
        font-weight: 700;
        color: #1b1b1b;
    }}
    /* --- Lưới Dashboard 2x2 --- */
    .dashboard-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        max-width: 1180px;
        margin: 0 auto;
    }}
    .chart-card {{
        background: #fff;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }}
    .chart-card h3 {{
        font-size: 15px;
        font-weight: 700;
        margin: 0 0 12px 0;
        color: #1b1b1b;
    }}
    .chart-card svg {{ width: 100%; height: auto; display: block; }}
    /* --- Layout riêng cho card Donut (biểu đồ + chú thích cạnh nhau) --- */
    .donut-wrapper {{
        display: flex;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
    }}
    .donut-wrapper svg {{ width: 220px; height: 220px; flex-shrink: 0; }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        margin-bottom: 8px;
        color: #333;
    }}
    .legend-swatch {{
        width: 13px;
        height: 13px;
        border-radius: 3px;
        display: inline-block;
        flex-shrink: 0;
    }}
    @media (max-width: 860px) {{
        .dashboard-grid, .kpi-row {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

    <div class="dashboard-header">
        <h1>DASHBOARD PHÂN TÍCH HIỆU QUẢ BÁN HÀNG - CỬA HÀNG E-COMMERCE (NĂM 2025)</h1>
        <p>Báo cáo tự động sinh bằng Python - Pandas - HTML/SVG</p>
    </div>

    <!-- KHỐI KPI EXECUTIVE SUMMARY -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Tổng Doanh Thu</div>
            <div class="kpi-value">{kpis['Tổng doanh thu (VNĐ)']:,.0f} đ</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Đơn Hàng Thành Công</div>
            <div class="kpi-value">{kpis['Tổng số đơn hàng thành công']:,.0f} đơn</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Giá Trị TB / Đơn (AOV)</div>
            <div class="kpi-value">{kpis['Giá trị trung bình / đơn (AOV - VNĐ)']:,.0f} đ</div>
        </div>
    </div>

    <!-- LƯỚI 4 BIỂU ĐỒ (GRID 2x2) -->
    <div class="dashboard-grid">

        <div class="chart-card">
            <h3>Xu Hướng Doanh Thu Theo Tháng</h3>
            {svg_line}
        </div>

        <div class="chart-card">
            <h3>Tổng Doanh Thu Theo Danh Mục Sản Phẩm</h3>
            {svg_bar}
        </div>

        <div class="chart-card">
            <h3>Tỷ Lệ Doanh Thu Theo Thành Phố</h3>
            <div class="donut-wrapper">
                {svg_donut}
                <div>{legend_donut}</div>
            </div>
        </div>

        <div class="chart-card">
            <h3>Tổng Số Lượng Sản Phẩm Bán Ra Theo Danh Mục</h3>
            {svg_hbar}
        </div>

    </div>

</body>
</html>"""

    # Lưu file HTML ngay tại thư mục chứa file script hiện tại
    output_html_path = OUTPUT_DIR / 'Dashboard_Ban_Hang_2025.html'
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Đã tạo Dashboard HTML tại: {output_html_path}")

    # Tự động mở file HTML vừa tạo bằng trình duyệt web mặc định
    webbrowser.open(output_html_path.resolve().as_uri())


# ==============================================================================
# PHẦN 5: XUẤT BÁO CÁO RA FILE CSV
# ==============================================================================
def export_report(df_clean: pd.DataFrame, output_path: str) -> None:
    """
    Xuất dữ liệu đã xử lý ra file CSV để lưu trữ/báo cáo.

    Tham số:
        df_clean (pd.DataFrame): Dữ liệu đã tiền xử lý.
        output_path (str): Đường dẫn file CSV đầu ra.
    """
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Đã xuất báo cáo chi tiết ra file: {output_path}")


# ==============================================================================
# PHẦN 6: HÀM CHÍNH - ĐIỀU PHỐI TOÀN BỘ QUY TRÌNH (MAIN PIPELINE)
# ==============================================================================
def main():
    """Hàm điều phối chạy toàn bộ pipeline phân tích dữ liệu từ A-Z."""

    print("Bước 1: Đang sinh dữ liệu giả lập...")
    df_raw = generate_sales_data(n_rows=800)
    print(f"  -> Đã sinh {len(df_raw)} dòng dữ liệu thô.\n")

    print("Bước 2: Đang tiền xử lý dữ liệu (lọc đơn hủy, tạo Doanh_Thu, Thang)...")
    df_clean = preprocess_data(df_raw)
    print(f"  -> Còn lại {len(df_clean)} dòng sau khi loại bỏ đơn 'Đã hủy'.\n")

    print("Bước 3: Đang tính toán các chỉ số KPI...")
    kpis = calculate_kpis(df_clean)
    print_executive_summary(kpis)

    print("Bước 4: Đang sinh Dashboard HTML/SVG và mở bằng trình duyệt...")
    create_dashboard(df_clean)
    print()

    print("Bước 5: Đang xuất báo cáo ra file CSV...")
    csv_output_path = OUTPUT_DIR / 'Bao_Cao_Ban_Hang_2025.csv'
    export_report(df_clean, csv_output_path)

    print("\nHOÀN TẤT PIPELINE PHÂN TÍCH DỮ LIỆU KINH DOANH!")


# Điểm khởi chạy chương trình
if __name__ == "__main__":
    main()
