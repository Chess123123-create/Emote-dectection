import tensorflow as tf # Thư viện Google, nền tảng chính để xây dựng AI
from tensorflow.keras.models import Sequential # Kiểu mô hình xếp chồng các lớp (Layer) lên nhau tuần tự
# Các lớp (Layers) quan trọng trong mạng CNN:
# - Conv2D: Tích chập, dùng để trích xuất đặc trưng (cạnh, góc, mắt, mũi...)
# - MaxPooling2D: Giảm kích thước ảnh, giữ lại đặc trưng quan trọng nhất (giảm tải tính toán)
# - Flatten: Duỗi ảnh từ ma trận 2D thành vector 1D để đưa vào lớp phân loại
# - Dense: Lớp nơ-ron kết nối đầy đủ (Fully Connected), đưa ra quyết định cuối cùng
# - Dropout: Ngắt ngẫu nhiên các nơ-ron để tránh học vẹt (Overfitting)
# - BatchNormalization: Chuẩn hóa dữ liệu giữa các lớp, giúp train nhanh và ổn định hơn
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Công cụ tăng cường dữ liệu ảnh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # Các công cụ hỗ trợ thông minh khi train
import os

# --- CẤU HÌNH HỆ THỐNG ---
# [LÝ THUYẾT] Tại sao 96x96?
# Ảnh gốc RAF-DB có chất lượng cao hơn FER2013 (48x48). 
# Tăng kích thước lên 96x96 giúp mạng CNN nhìn rõ các biểu cảm vi mô (như nheo mắt, nhếch mép).
IMG_SIZE = 96        
BATCH_SIZE = 32      # Số lượng ảnh được đưa vào GPU/CPU học cùng lúc. 32 là con số chuẩn cho các máy tính cá nhân.
EPOCHS = 50          # Số vòng lặp học lại toàn bộ dữ liệu.
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/validation'

# Kiểm tra đường dẫn tồn tại để tránh lỗi crash chương trình
if not os.path.exists(TRAIN_DIR):
    print(f"❌ LỖI: Không tìm thấy thư mục '{TRAIN_DIR}'")
    exit()

# --- 1. CHUẨN BỊ DỮ LIỆU (DATA PREPROCESSING & AUGMENTATION) ---

# [KỸ THUẬT XỬ LÝ ẢNH] Data Augmentation (Tăng cường dữ liệu)
# Vấn đề: Nếu chỉ học ảnh thẳng, khi người dùng nghiêng đầu, AI sẽ không nhận ra.
# Giải pháp: ImageDataGenerator tự động tạo ra các biến thể của ảnh gốc trong lúc train.
train_datagen = ImageDataGenerator(
    rescale=1./255,         # [QUAN TRỌNG] Chuẩn hóa pixel từ [0-255] về [0-1] giúp tính toán nhanh hơn.
    rotation_range=20,      # Xoay ảnh ngẫu nhiên tối đa 20 độ.
    width_shift_range=0.1,  # Dịch chuyển sang trái/phải.
    height_shift_range=0.1, # Dịch chuyển lên/xuống.
    shear_range=0.1,        # Làm méo ảnh (như nhìn nghiêng).
    zoom_range=0.1,         # Phóng to/thu nhỏ ngẫu nhiên.
    horizontal_flip=True,   # Lật gương (Ví dụ: Cười nghiêng trái lật thành nghiêng phải vẫn là cười).
    fill_mode='nearest'     # Điền đầy các pixel bị khuyết khi xoay ảnh bằng pixel lân cận.
)

# Tập kiểm thử (Validation) chỉ cần chuẩn hóa, TUYỆT ĐỐI KHÔNG xoay/lật để đánh giá công bằng.
val_datagen = ImageDataGenerator(rescale=1./255)

print("--> Đang load dữ liệu từ ổ cứng...")

# [CƠ CHẾ LOAD DỮ LIỆU] flow_from_directory
# Load ảnh theo từng lô (Batch) thay vì load tất cả vào RAM (tránh tràn RAM).
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE), # Resize toàn bộ ảnh về 96x96
    color_mode='rgb',       # [QUAN TRỌNG] RAF-DB là ảnh màu, dùng 3 kênh (RGB) chứa nhiều thông tin cảm xúc hơn ảnh xám.
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Phân loại nhiều lớp (7 cảm xúc) -> Dùng one-hot encoding.
    shuffle=True            # Trộn ngẫu nhiên ảnh để model không học thuộc thứ tự.
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',       # Phải khớp với tập train.
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- 2. XÂY DỰNG MODEL (CNN ARCHITECTURE) ---

# Nhóm thiết kế mạng CNN theo phong cách VGG (Visual Geometry Group) nhưng thu nhỏ.
# Quy tắc hình nón: Càng vào sâu, kích thước ảnh (Height, Width) càng nhỏ, nhưng độ sâu (Filters) càng tăng.
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)), # Đầu vào: Ảnh 96x96, 3 kênh màu (RGB)
    
    # --- Block 1: Trích xuất đặc điểm mức thấp (Low-level features) ---
    # Tìm các đường nét, cạnh, góc, màu sắc cơ bản.
    Conv2D(32, (3, 3), activation='relu', padding='same'), # 32 bộ lọc
    BatchNormalization(), # Giữ dữ liệu ổn định
    MaxPooling2D(2, 2),   # Giảm kích thước ảnh đi một nửa (96->48)
    Dropout(0.25),        # Quên bớt 25% thông tin để tránh học vẹt

    # --- Block 2: Trích xuất đặc điểm trung cấp (Mid-level features) ---
    # Tìm các hình dạng: mắt, mũi, miệng, lông mày.
    Conv2D(64, (3, 3), activation='relu', padding='same'), # Tăng lên 64 bộ lọc
    BatchNormalization(),
    MaxPooling2D(2, 2),   # Giảm kích thước (48->24)
    Dropout(0.25),

    # --- Block 3: Trích xuất đặc điểm cao cấp (High-level features) ---
    # Kết hợp mắt, mũi, miệng để nhận ra "khuôn mặt đang cười" hay "đang khóc".
    Conv2D(128, (3, 3), activation='relu', padding='same'), # Tăng lên 128 bộ lọc
    BatchNormalization(),
    MaxPooling2D(2, 2),   # Giảm kích thước (24->12)
    Dropout(0.25),
    
    # --- Block 4: Trừu tượng hóa cao độ (Deep semantic features) ---
    # Cần thiết vì ảnh input lớn (96x96), cần thêm tầng để xử lý sâu hơn.
    Conv2D(256, (3, 3), activation='relu', padding='same'), # Tăng lên 256 bộ lọc
    BatchNormalization(),
    MaxPooling2D(2, 2),   # Giảm kích thước (12->6)
    Dropout(0.25),

    # --- Phân loại (Classification Head) ---
    Flatten(), # Duỗi khối 3D (6x6x256) thành 1 vector dài để tính toán xác suất.
    Dense(512, activation='relu'), # Lớp nơ-ron dày đặc suy luận logic.
    BatchNormalization(),
    Dropout(0.5), # Quên 50% ở lớp cuối cực kỳ quan trọng để model tổng quát hóa tốt.
    
    # Output Layer: 7 nơ-ron tương ứng 7 cảm xúc.
    # Hàm Softmax: Chuyển đầu ra thành xác suất % (VD: Vui 80%, Buồn 20%).
    Dense(7, activation='softmax') 
])

# [LÝ THUYẾT] Optimizer & Loss Function
# - Adam: Thuật toán tối ưu phổ biến nhất, tự điều chỉnh tốc độ học.
# - Categorical Crossentropy: Hàm mất mát chuẩn cho bài toán phân loại nhiều lớp (Multi-class).
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# --- 3. CÁC CÔNG CỤ HỖ TRỢ (CALLBACKS) ---
# Đây là các "trợ lý" giúp quá trình train thông minh hơn.

# 1. ModelCheckpoint: "Lưu lại khoảnh khắc huy hoàng nhất"
# Chỉ lưu model khi 'val_accuracy' (độ chính xác trên tập kiểm thử) đạt đỉnh mới.
# Giúp ta lấy được model tốt nhất (Epoch 38) chứ không phải model cuối cùng (Epoch 50).
checkpoint = ModelCheckpoint(
    'best_rafdb_model.keras',  
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max',
    verbose=1
)

# 2. ReduceLROnPlateau: "Học chậm lại khi gặp khó"
# Nếu sau 3 vòng (patience=3) mà loss không giảm, nó tự động giảm tốc độ học (factor=0.2).
# Giống như việc ta đi chậm lại khi dò đường vào ngõ hẹp để tìm đích chính xác hơn.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=0.00001, 
    verbose=1
)

# 3. EarlyStopping: "Dừng lại khi không còn tiến bộ"
# Nếu sau 8 vòng (patience=8) mà độ chính xác không tăng, dừng train ngay lập tức.
# Giúp tiết kiệm thời gian và điện năng, tránh overfitting.
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=8, 
    restore_best_weights=True, 
    verbose=1
)

# --- 4. BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN (TRAINING) ---
# [Quy trình chạy của hàm fit]:
# 1. Forward Pass: Đưa ảnh qua các lớp Conv -> Dense -> Dự đoán.
# 2. Loss Calculation: So sánh dự đoán với nhãn gốc -> Tính sai số (Loss).
# 3. Backpropagation: Lan truyền ngược sai số để điều chỉnh trọng số (Weights) của mạng.
# 4. Lặp lại cho đến khi hết Epochs hoặc EarlyStopping kích hoạt.
print(f"--> Bắt đầu Train trên {train_generator.samples} ảnh...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Tính số bước trong 1 epoch
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# --- 5. LƯU MODEL CUỐI CÙNG ---
# Lưu thêm bản .h5 truyền thống để tương thích ngược với các code cũ nếu cần.
model.save('my_rgb_model.h5')
print("--> File 'my_rgb_model.h5' đã sẵn sàng.")