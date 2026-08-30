import pandas as pd
import matplotlib.pyplot as plt

# 1. Tạo dữ liệu mẫu dạng Dictionary
data = {
    'Thang': ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6'],
    'Doanh_Thu': [150, 180, 220, 200, 280, 310],     # Đơn vị: Triệu VNĐ
    'Chi_Phi': [100, 110, 130, 140, 160, 170],      # Đơn vị: Triệu VNĐ
    'DanhMuc_Chinh': ['Điện thoại', 'Điện thoại', 'Laptop', 'Laptop', 'Phụ kiện', 'Phụ kiện']
}

# 2. Chuyển đổi thành DataFrame (Bảng dữ liệu của Pandas)
df = pd.DataFrame(data)
print("--- BẢNG DỮ LIỆU BAN ĐẦU ---")
print(df)
# Tính Lợi nhuận = Doanh thu - Chi phí
df['Loi_Nhuan'] = df['Doanh_Thu'] - df['Chi_Phi']

# Tính các chỉ số thống kê
tong_doanh_thu = df['Doanh_Thu'].sum()
loi_nhuan_trung_binh = df['Loi_Nhuan'].mean()

print("\n--- BẢNG DỮ LIỆU SAU KHI BỔ SUNG LỢI NHUẬN ---")
print(df[['Thang', 'Doanh_Thu', 'Chi_Phi', 'Loi_Nhuan']])
print(f"\nTong Doanh Thu: {tong_doanh_thu} Triệu VNĐ")
print(f"Loi Nhuan Trung Binh / Thang: {loi_nhuan_trung_binh:.1f} Triệu VNĐ")
# Cấu hình kích thước khung hình
plt.figure(figsize=(10, 5))

# Vẽ biểu đồ Doanh thu và Chi phí (dạng cột)
plt.bar(df['Thang'], df['Doanh_Thu'], label='Doanh Thu', alpha=0.6, color='skyblue')
plt.bar(df['Thang'], df['Chi_Phi'], label='Chi Phí', alpha=0.6, color='orange')

# Vẽ biểu đồ Lợi nhuận (dạng đường)
plt.plot(df['Thang'], df['Loi_Nhuan'], label='Lợi Nhuận', color='green', marker='o', linewidth=2.5)

# Thêm tiêu đề và nhãn các trục
plt.title('BÁO CÁO TÌNH HÌNH KINH DOANH 6 THÁNG ĐẦU NĂM', fontsize=14, fontweight='bold')
plt.xlabel('Thời Gian')
plt.ylabel('Giá Trị (Triệu VNĐ)')

# Thêm lưới, chú thích và hiển thị
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
# Gom nhóm theo Danh mục sản phẩm và tính tổng
df_danh_muc = df.groupby('DanhMuc_Chinh')[['Doanh_Thu', 'Loi_Nhuan']].sum()

print("\n--- TỔNG KẾT THEO DANH MỤC ---")
print(df_danh_muc)

# Vẽ biểu đồ tròn thể hiện tỷ lệ lợi nhuận theo danh mục
plt.figure(figsize=(6, 6))
plt.pie(
    df_danh_muc['Loi_Nhuan'], 
    labels=df_danh_muc.index, 
    autopct='%1.1f%%', 
    startangle=140,
    colors=['#ff9999','#66b3ff','#99ff99']
)
plt.title('TỶ LỆ LỢI NHUẬN THEO DANH MỤC SẢN PHẨM', fontsize=12, fontweight='bold')
plt.show()