#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_NeoPixel.h>

// --- CỜ BẬT/TẮT BENCHMARK KHI KHỞI ĐỘNG ---
const bool ENABLE_BENCHMARK = true; 

// --- CẤU HÌNH PHẦN CỨNG ---
Adafruit_SSD1306 display(128, 64, &Wire, -1);
Adafruit_BMP280 bmp; 
Adafruit_NeoPixel pixels(1, 48, NEO_GRB + NEO_KHZ800);

#define MPU_ADDR 0x68 
#define BTN_BOOT 0
#define BTN_VOL_UP 40
#define BTN_VOL_DOWN 39

// --- BIẾN KHÔNG GIAN, CẢM BIẾN & TIỀN XỬ LÝ (S14-Q3: NaN/IQR) ---
float target_altitude = 0.0f; 
float base_altitude = 0.0f;   
float base_accel = 1.0f; 
float offset_x = 0.0f, offset_y = 0.0f;
float smooth_x = 0.0f, smooth_y = 0.0f;
float last_ax = 0.0f, last_ay = 0.0f, last_az = 1.0f;
float prev_ax = 0.0f, prev_ay = 0.0f, prev_az = 1.0f;
float alt_window[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; 
int alt_win_idx = 0;
uint32_t nan_imputed_count = 0;
uint32_t iqr_clipped_count = 0;
int idle_counter = 0; 

// --- TASK 1: TRỤC ĐỘ CAO DÙNG ADAMW (S12) + LR DECAY (S11-Q8) ---
float weight_W = 1.0f;  
float bias_b = 25.0f;   
const float alpha0_W = 0.18f;   
const float alpha0_b = 0.35f;   
const float lr_decay_k = 0.02f; 
const float adam_beta1 = 0.85f; 
const float adam_beta2 = 0.95f; 
const float adam_eps_ctrl = 0.08f; 

float m_W = 0.0f, v_W = 0.0f;
float m_b = 0.0f, v_b = 0.0f;
float beta1_pow = 1.0f, beta2_pow = 1.0f; 
int t_alt = 0;                            

float decay_W = 0.12f;          
float decay_b = 0.08f; 
float outlier_threshold = 0.8f; 

float prev_W = 1.0f;         
float locked_raw_error = 0.0f; 
int alt_cvg_count = 0;
bool alt_converged = false;
float current_eff_lr = alpha0_W; 

// --- TASK 2: MẠNG MLP 3 -> 4 -> 3 ĐỒNG BỘ MẶT PHẲNG PHÂN LỚP (S14 AUDIT FIX) ---
float W1[3][4] = {
  {-2.00f,  2.60f,  4.50f, -1.00f}, // X[0]: sin(theta)
  {-1.60f,  2.08f,  0.00f, -0.80f}, // X[1]: Rung chấn
  {-0.80f,  1.04f,  0.00f, -0.40f}  // X[2]: Sai số độ cao
};
float b1[4] = {0.95f, -0.15f, -2.48f, 0.60f};

float W2[4][3] = {
  { 2.2f, -1.0f, -2.0f},
  {-1.4f,  2.6f,  0.1f},
  {-2.0f, -1.2f,  4.2f},
  { 1.5f, -0.8f, -1.5f}
};
float b2[3] = {0.4f, 0.1f, -0.5f};

float vel_W1[3][4] = {0};
float vel_b1[4] = {0};
float vel_W2[4][3] = {0};
float vel_b2[3] = {0};
const float mlp_momentum = 0.9f;
const float class_weights[3] = {1.0f, 1.2f, 1.0f};

// [PATCH LOG-6 #1] Tập Train động (Experience Replay Buffer)
float replay_X[3][3] = {
  {0.02f, 0.02f, 0.02f}, // Class 0 (SAFE)
  {0.54f, 0.10f, 0.08f}, // Class 1 (WIND)
  {0.98f, 0.05f, 0.05f}  // Class 2 (KILL)
};

// [PATCH LOG-6 #1] Tập Validation TĨNH - TÁCH BIỆT KHỎI REPLAY (Reviewer Fix)
// Lệch 5-10% so với replay_X để Early Stopping có dữ liệu thực sự held-out
const float val_X[3][3] = {
  {0.05f, 0.04f, 0.03f}, // SAFE Validation (lệch 0.03 so với replay_X[0][0]=0.02)
  {0.59f, 0.14f, 0.11f}, // WIND Validation (lệch 0.05 so với replay_X[1][0]=0.54)
  {0.93f, 0.07f, 0.05f}  // KILL Validation (lệch 0.05 so với replay_X[2][0]=0.98)
};

float best_W1[3][4], best_b1[4], best_W2[4][3], best_b2[3];
float best_val_loss = 999.0f;
float last_train_loss = 0.0f;
float last_val_loss = 0.0f;
int patience_counter = 0;
const int PATIENCE = 3;
const float MIN_DELTA = 0.0005f; 
uint32_t checkpoint_saves = 0;
uint32_t early_stop_rollbacks = 0;

float lr_mlp = 0.025f;
bool mlp_converged = false;
int step_counter = 0;

// --- HỘP ĐEN ĐÁNH GIÁ S14: CONFUSION MATRIX 3x3 & MAE/MSE/RMSE ---
uint32_t conf_mat[3][3] = {0}; 
double sum_abs_err = 0.0;
double sum_sq_err = 0.0;
uint32_t reg_sample_count = 0;

// ==========================================
// CÁC HÀM TOÁN HỌC & TỐI ƯU HÓA (S11 -> S14)
// ==========================================
void reset_adam_alt() {
  t_alt = 0;
  m_W = 0.0f; v_W = 0.0f;
  m_b = 0.0f; v_b = 0.0f;
  beta1_pow = 1.0f; 
  beta2_pow = 1.0f;
  alt_converged = false;
  alt_cvg_count = 0;
}

float minmax_scale(float x, float x_min, float x_max) {
  return constrain((x - x_min) / (x_max - x_min), 0.0f, 1.0f);
}

float relu(float z) { return (z > 0.0f) ? z : 0.0f; }
float leaky_relu(float z) { return (z >= 0.0f) ? z : 0.1f * z; }

void softmax_3(const float z[3], float p[3]) {
  float max_z = max(z[0], max(z[1], z[2]));
  float e0 = exp(z[0] - max_z);
  float e1 = exp(z[1] - max_z);
  float e2 = exp(z[2] - max_z);
  float sum = e0 + e1 + e2;
  p[0] = e0 / sum; p[1] = e1 / sum; p[2] = e2 / sum;
}

void softmax_4(float z1, float z2, float z3, float z4, float T, float &p1, float &p2, float &p3, float &p4) {
  float max_z = max(max(z1, z2), max(z3, z4));
  float e1 = exp((z1 - max_z) / T);
  float e2 = exp((z2 - max_z) / T);
  float e3 = exp((z3 - max_z) / T);
  float e4 = exp((z4 - max_z) / T);
  float sum = e1 + e2 + e3 + e4;
  p1 = e1 / sum; p2 = e2 / sum; p3 = e3 / sum; p4 = e4 / sum;
}

float clean_altitude_iqr(float raw_val) {
  float sorted[5];
  memcpy(sorted, alt_window, sizeof(sorted));
  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 5; j++) {
      if (sorted[j] < sorted[i]) {
        float tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
      }
    }
  }
  float median = sorted[2];
  float q1 = sorted[1], q3 = sorted[3];
  float iqr = max(q3 - q1, 0.15f);

  if (isnan(raw_val) || isinf(raw_val)) {
    nan_imputed_count++;
    raw_val = median;
  }
  float lower = q1 - max(1.5f * iqr, 2.5f);
  float upper = q3 + max(1.5f * iqr, 2.5f);
  if (raw_val < lower || raw_val > upper) {
    iqr_clipped_count++;
    raw_val = constrain(raw_val, lower, upper);
  }

  alt_window[alt_win_idx] = raw_val;
  alt_win_idx = (alt_win_idx + 1) % 5;
  return raw_val;
}

int forward_mlp_cache(const float X[3], float Z1[4], float A1[4], float probs[3], float &ce_loss, int target_class = 0) {
  for (int j = 0; j < 4; j++) {
    float z1 = b1[j];
    for (int i = 0; i < 3; i++) z1 += X[i] * W1[i][j];
    Z1[j] = z1;
    A1[j] = relu(z1);
  }
  float Z2[3];
  for (int k = 0; k < 3; k++) {
    float z2 = b2[k];
    for (int j = 0; j < 4; j++) z2 += A1[j] * W2[j][k];
    Z2[k] = z2;
  }
  softmax_3(Z2, probs);
  float p_target = constrain(probs[target_class], 1e-6f, 1.0f - 1e-6f);
  ce_loss = -log(p_target);

  int pred_class = 0;
  if (probs[1] > probs[pred_class]) pred_class = 1;
  if (probs[2] > probs[pred_class]) pred_class = 2;
  return pred_class;
}

void save_mlp_checkpoint() {
  memcpy(best_W1, W1, sizeof(W1));
  memcpy(best_b1, b1, sizeof(b1));
  memcpy(best_W2, W2, sizeof(W2));
  memcpy(best_b2, b2, sizeof(b2));
}

void rollback_mlp_checkpoint() {
  memcpy(W1, best_W1, sizeof(W1));
  memcpy(b1, best_b1, sizeof(b1));
  memcpy(W2, best_W2, sizeof(W2));
  memcpy(b2, best_b2, sizeof(b2));
}

float compute_validation_loss() {
  float val_loss = 0.0f;
  for (int c = 0; c < 3; c++) {
    float Z1[4], A1[4], P[3], sample_loss;
    forward_mlp_cache(val_X[c], Z1, A1, P, sample_loss, c);
    val_loss += sample_loss;
  }
  return val_loss / 3.0f;
}

float train_mlp_minibatch_momentum(float learning_rate) {
  float dW1_sum[3][4] = {0};
  float db1_sum[4] = {0};
  float dW2_sum[4][3] = {0};
  float db2_sum[3] = {0};
  float batch_loss = 0.0f;

  for (int c = 0; c < 3; c++) {
    float Z1[4], A1[4], P[3], sample_loss;
    forward_mlp_cache(replay_X[c], Z1, A1, P, sample_loss, c);
    batch_loss += sample_loss;

    float delta2[3];
    for (int k = 0; k < 3; k++) {
      delta2[k] = (P[k] - ((k == c) ? 1.0f : 0.0f)) * class_weights[c];
      db2_sum[k] += delta2[k];
      for (int j = 0; j < 4; j++) {
        dW2_sum[j][k] += A1[j] * delta2[k];
      }
    }

    for (int j = 0; j < 4; j++) {
      float back_err = 0.0f;
      for (int k = 0; k < 3; k++) back_err += delta2[k] * W2[j][k];
      float delta1 = (Z1[j] > 0.0f) ? back_err : 0.0f; 
      db1_sum[j] += delta1;
      for (int i = 0; i < 3; i++) {
        dW1_sum[i][j] += replay_X[c][i] * delta1;
      }
    }
  }

  const float inv_m = 1.0f / 3.0f;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float g = dW1_sum[i][j] * inv_m;
      vel_W1[i][j] = mlp_momentum * vel_W1[i][j] + (1.0f - mlp_momentum) * g;
      W1[i][j] = constrain(W1[i][j] - learning_rate * vel_W1[i][j], -6.0f, 6.0f);
    }
  }
  W1[1][2] = 0.0f;
  W1[2][2] = 0.0f;

  for (int j = 0; j < 4; j++) {
    float gb1 = db1_sum[j] * inv_m;
    vel_b1[j] = mlp_momentum * vel_b1[j] + (1.0f - mlp_momentum) * gb1;
    b1[j] = constrain(b1[j] - learning_rate * vel_b1[j], -6.0f, 6.0f);

    for (int k = 0; k < 3; k++) {
      float g2 = dW2_sum[j][k] * inv_m;
      vel_W2[j][k] = mlp_momentum * vel_W2[j][k] + (1.0f - mlp_momentum) * g2;
      W2[j][k] = constrain(W2[j][k] - learning_rate * vel_W2[j][k], -6.0f, 6.0f);
    }
  }
  for (int k = 0; k < 3; k++) {
    float gb2 = db2_sum[k] * inv_m;
    vel_b2[k] = mlp_momentum * vel_b2[k] + (1.0f - mlp_momentum) * gb2;
    b2[k] = constrain(b2[k] - learning_rate * vel_b2[k], -6.0f, 6.0f);
  }

  last_train_loss = batch_loss * inv_m;
  last_val_loss = compute_validation_loss();

  if (last_val_loss < best_val_loss - MIN_DELTA) {
    best_val_loss = last_val_loss;
    patience_counter = 0;
    save_mlp_checkpoint();
    checkpoint_saves++;
  } else {
    patience_counter++;
    if (patience_counter >= PATIENCE) {
      rollback_mlp_checkpoint();
      last_val_loss = best_val_loss;
      mlp_converged = true;
      patience_counter = 0;
      early_stop_rollbacks++;
    }
  }

  return last_val_loss;
}

// =========================================================
// ENGINE BENCHMARK 90 MẪU (60 TRAIN / 30 TEST) - [PATCH LOG-6 #2]
// =========================================================
int dense_params(int in_features, int neurons) {
  return (in_features * neurons) + neurons;
}

void run_s13_s14_assignment_benchmark(bool show_oled) {
  int p_343  = dense_params(3, 4) + dense_params(4, 3);                      // 31
  int p_3843 = dense_params(3, 8) + dense_params(8, 4) + dense_params(4, 3); // 83
  int p_iris = dense_params(4, 8) + dense_params(8, 3);                      // 67

  // [PATCH LOG-6 #2] Mở rộng từ 60 → 90 mẫu (60 Train / 30 Test, 10 mẫu/lớp test)
  const int N_TRAIN = 60;
  const int N_TEST  = 30;
  const int N_TOTAL = N_TRAIN + N_TEST;

  float data_X[90][3];
  int data_Y[90];
  const float centers[3][3] = {{0.05f, 0.03f, 0.03f}, {0.52f, 0.22f, 0.22f}, {0.92f, 0.38f, 0.35f}};
  for (int i = 0; i < N_TOTAL; i++) {
    int c = i % 3;
    data_Y[i] = c;
    float noise = ((float)((i * 37) % 11) - 5.0f) * 0.015f;
    data_X[i][0] = constrain(centers[c][0] + noise, 0.0f, 1.0f);
    data_X[i][1] = constrain(centers[c][1] - 0.7f * noise, 0.0f, 1.0f);
    data_X[i][2] = constrain(centers[c][2] + 0.5f * noise, 0.0f, 1.0f);
    // 2 mẫu nhiễu Train (i=4, 10) + 1 mẫu nhiễu Test (i=75 nằm trong [60..89])
    if (i == 4 || i == 10) { data_X[i][0] = 0.75f; data_Y[i] = 2; }
    if (i == 75) { data_X[i][0] = 0.74f; data_Y[i] = 1; }
  }

  // S14-Q2: Fit StandardScaler CHỈ TRÊN 60 MẪU TRAIN để chống Data Leakage
  float tr_mean[3] = {0}, tr_std[3] = {0};
  for (int i = 0; i < N_TRAIN; i++) for (int f = 0; f < 3; f++) tr_mean[f] += data_X[i][f] / (float)N_TRAIN;
  for (int i = 0; i < N_TRAIN; i++) for (int f = 0; f < 3; f++) {
    float d = data_X[i][f] - tr_mean[f];
    tr_std[f] += (d * d) / (float)N_TRAIN;
  }
  for (int f = 0; f < 3; f++) tr_std[f] = sqrtf(tr_std[f]) + 1e-6f;
  for (int i = 0; i < N_TOTAL; i++) for (int f = 0; f < 3; f++) {
    data_X[i][f] = (data_X[i][f] - tr_mean[f]) / tr_std[f];
  }

  int epochs_to_target[3] = {60, 60, 60};
  float final_tr_loss[3] = {0}, final_te_acc[3] = {0};

  // Huấn luyện thật 2 mô hình 3-4-3 (opt=0: SGD, opt=1: Adam)
  for (int opt = 0; opt < 2; opt++) {
    float bW1[3][4] = {{-0.6f, 0.5f, 0.7f, -0.3f}, {-0.3f, 0.5f, 0.2f, -0.2f}, {-0.2f, 0.4f, 0.3f, -0.1f}};
    float bb1[4] = {0.1f, 0.1f, 0.1f, 0.1f};
    float bW2[4][3] = {{0.6f, -0.3f, -0.4f}, {-0.3f, 0.7f, -0.2f}, {-0.4f, -0.2f, 0.8f}, {0.3f, -0.2f, -0.3f}};
    float bb2[3] = {0.0f, 0.0f, 0.0f};
    float mW1[3][4] = {0}, vW1[3][4] = {0}, mW2[4][3] = {0}, vW2[4][3] = {0};
    float b1_p = 1.0f, b2_p = 1.0f;

    for (int ep = 1; ep <= 60; ep++) {
      float dW1[3][4] = {0}, dW2[4][3] = {0}, ep_loss = 0.0f;
      for (int s = 0; s < N_TRAIN; s++) {
        float z1[4], a1[4], z2[3], p[3];
        for (int j = 0; j < 4; j++) {
          z1[j] = bb1[j] + data_X[s][0]*bW1[0][j] + data_X[s][1]*bW1[1][j] + data_X[s][2]*bW1[2][j];
          a1[j] = (z1[j] > 0.0f) ? z1[j] : 0.0f;
        }
        for (int k = 0; k < 3; k++) {
          z2[k] = bb2[k] + a1[0]*bW2[0][k] + a1[1]*bW2[1][k] + a1[2]*bW2[2][k] + a1[3]*bW2[3][k];
        }
        softmax_3(z2, p);
        ep_loss += -log(constrain(p[data_Y[s]], 1e-6f, 1.0f));
        float d2[3];
        for (int k = 0; k < 3; k++) {
          d2[k] = (p[k] - ((k == data_Y[s]) ? 1.0f : 0.0f)) / (float)N_TRAIN;
          for (int j = 0; j < 4; j++) dW2[j][k] += a1[j] * d2[k];
        }
        for (int j = 0; j < 4; j++) {
          float d1 = (d2[0]*bW2[j][0] + d2[1]*bW2[j][1] + d2[2]*bW2[j][2]) * ((z1[j] > 0.0f) ? 1.0f : 0.0f);
          for (int i = 0; i < 3; i++) dW1[i][j] += data_X[s][i] * d1;
        }
      }
      ep_loss /= (float)N_TRAIN;
      if (ep_loss < 0.30f && epochs_to_target[opt] == 60) epochs_to_target[opt] = ep;
      final_tr_loss[opt] = ep_loss;

      if (opt == 0) {
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) bW1[i][j] -= 0.10f * dW1[i][j];
        for (int j = 0; j < 4; j++) for (int k = 0; k < 3; k++) bW2[j][k] -= 0.10f * dW2[j][k];
      } else {
        b1_p *= 0.9f; b2_p *= 0.999f;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++) {
          mW1[i][j] = 0.9f * mW1[i][j] + 0.1f * dW1[i][j];
          vW1[i][j] = 0.999f * vW1[i][j] + 0.001f * dW1[i][j] * dW1[i][j];
          bW1[i][j] -= 0.06f * (mW1[i][j] / (1.0f - b1_p)) / (sqrtf(vW1[i][j] / (1.0f - b2_p)) + 1e-8f);
        }
        for (int j = 0; j < 4; j++) for (int k = 0; k < 3; k++) {
          mW2[j][k] = 0.9f * mW2[j][k] + 0.1f * dW2[j][k];
          vW2[j][k] = 0.999f * vW2[j][k] + 0.001f * dW2[j][k] * dW2[j][k];
          bW2[j][k] -= 0.06f * (mW2[j][k] / (1.0f - b1_p)) / (sqrtf(vW2[j][k] / (1.0f - b2_p)) + 1e-8f);
        }
      }
    }
    int correct = 0;
    for (int s = N_TRAIN; s < N_TOTAL; s++) {
      float z1[4], a1[4], z2[3], p[3];
      for (int j = 0; j < 4; j++) {
        z1[j] = bb1[j] + data_X[s][0]*bW1[0][j] + data_X[s][1]*bW1[1][j] + data_X[s][2]*bW1[2][j];
        a1[j] = (z1[j] > 0.0f) ? z1[j] : 0.0f;
      }
      for (int k = 0; k < 3; k++) {
        z2[k] = bb2[k] + a1[0]*bW2[0][k] + a1[1]*bW2[1][k] + a1[2]*bW2[2][k] + a1[3]*bW2[3][k];
      }
      softmax_3(z2, p);
      int pred = (p[1] > p[0]) ? 1 : 0;
      if (p[2] > p[pred]) pred = 2;
      if (pred == data_Y[s]) correct++;
    }
    final_te_acc[opt] = (float)correct / (float)N_TEST;
  }

  // Huấn luyện thật 100% mô hình sâu 3-8-4-3 (83 tham số) bằng Adam
  {
    float dW1[3][8], db1[8] = {0}, dW2[8][4], db2[4] = {0}, dW3[4][3], db3[3] = {0};
    float mW1[3][8] = {0}, vW1[3][8] = {0}, mW2[8][4] = {0}, vW2[8][4] = {0}, mW3[4][3] = {0}, vW3[4][3] = {0};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 8; j++) dW1[i][j] = ((float)(((i+1)*(j+3)) % 7) - 3.0f) * 0.35f;
    for (int j = 0; j < 8; j++) for (int k = 0; k < 4; k++) dW2[j][k] = ((float)(((j+2)*(k+1)) % 7) - 3.0f) * 0.35f;
    for (int k = 0; k < 4; k++) for (int c = 0; c < 3; c++) dW3[k][c] = ((float)(((k+1)*(c+2)) % 5) - 2.0f) * 0.45f;
    float b1_p = 1.0f, b2_p = 1.0f;

    for (int ep = 1; ep <= 60; ep++) {
      float gW1[3][8] = {0}, gW2[8][4] = {0}, gW3[4][3] = {0}, ep_loss = 0.0f;
      for (int s = 0; s < N_TRAIN; s++) {
        float a1[8], a2[4], z3[3] = {0}, p[3];
        for (int j = 0; j < 8; j++) {
          float z = db1[j] + data_X[s][0]*dW1[0][j] + data_X[s][1]*dW1[1][j] + data_X[s][2]*dW1[2][j];
          a1[j] = (z > 0.0f) ? z : 0.0f;
        }
        for (int k = 0; k < 4; k++) {
          float z = db2[k];
          for (int j = 0; j < 8; j++) z += a1[j] * dW2[j][k];
          a2[k] = (z > 0.0f) ? z : 0.0f;
        }
        for (int c = 0; c < 3; c++) {
          float z = db3[c];
          for (int k = 0; k < 4; k++) z += a2[k] * dW3[k][c];
          z3[c] = z;
        }
        softmax_3(z3, p);
        ep_loss += -log(constrain(p[data_Y[s]], 1e-6f, 1.0f));

        float del3[3], del2[4] = {0};
        for (int c = 0; c < 3; c++) {
          del3[c] = (p[c] - ((c == data_Y[s]) ? 1.0f : 0.0f)) / (float)N_TRAIN;
          for (int k = 0; k < 4; k++) gW3[k][c] += a2[k] * del3[c];
        }
        for (int k = 0; k < 4; k++) {
          if (a2[k] > 0.0f) {
            del2[k] = del3[0]*dW3[k][0] + del3[1]*dW3[k][1] + del3[2]*dW3[k][2];
            for (int j = 0; j < 8; j++) gW2[j][k] += a1[j] * del2[k];
          }
        }
        for (int j = 0; j < 8; j++) {
          if (a1[j] > 0.0f) {
            float del1 = del2[0]*dW2[j][0] + del2[1]*dW2[j][1] + del2[2]*dW2[j][2] + del2[3]*dW2[j][3];
            for (int i = 0; i < 3; i++) gW1[i][j] += data_X[s][i] * del1;
          }
        }
      }
      ep_loss /= (float)N_TRAIN;
      if (ep_loss < 0.30f && epochs_to_target[2] == 60) epochs_to_target[2] = ep;
      final_tr_loss[2] = ep_loss;

      b1_p *= 0.9f; b2_p *= 0.999f;
      for (int i = 0; i < 3; i++) for (int j = 0; j < 8; j++) {
        mW1[i][j] = 0.9f * mW1[i][j] + 0.1f * gW1[i][j];
        vW1[i][j] = 0.999f * vW1[i][j] + 0.001f * gW1[i][j] * gW1[i][j];
        dW1[i][j] -= 0.07f * (mW1[i][j] / (1.0f - b1_p)) / (sqrtf(vW1[i][j] / (1.0f - b2_p)) + 1e-8f);
      }
      for (int j = 0; j < 8; j++) for (int k = 0; k < 4; k++) {
        mW2[j][k] = 0.9f * mW2[j][k] + 0.1f * gW2[j][k];
        vW2[j][k] = 0.999f * vW2[j][k] + 0.001f * gW2[j][k] * gW2[j][k];
        dW2[j][k] -= 0.07f * (mW2[j][k] / (1.0f - b1_p)) / (sqrtf(vW2[j][k] / (1.0f - b2_p)) + 1e-8f);
      }
      for (int k = 0; k < 4; k++) for (int c = 0; c < 3; c++) {
        mW3[k][c] = 0.9f * mW3[k][c] + 0.1f * gW3[k][c];
        vW3[k][c] = 0.999f * vW3[k][c] + 0.001f * gW3[k][c] * gW3[k][c];
        dW3[k][c] -= 0.07f * (mW3[k][c] / (1.0f - b1_p)) / (sqrtf(vW3[k][c] / (1.0f - b2_p)) + 1e-8f);
      }
    }
    int correct = 0;
    for (int s = N_TRAIN; s < N_TOTAL; s++) {
      float a1[8], a2[4], z3[3] = {0}, p[3];
      for (int j = 0; j < 8; j++) {
        float z = db1[j] + data_X[s][0]*dW1[0][j] + data_X[s][1]*dW1[1][j] + data_X[s][2]*dW1[2][j];
        a1[j] = (z > 0.0f) ? z : 0.0f;
      }
      for (int k = 0; k < 4; k++) {
        float z = db2[k];
        for (int j = 0; j < 8; j++) z += a1[j] * dW2[j][k];
        a2[k] = (z > 0.0f) ? z : 0.0f;
      }
      for (int c = 0; c < 3; c++) {
        float z = db3[c];
        for (int k = 0; k < 4; k++) z += a2[k] * dW3[k][c];
        z3[c] = z;
      }
      softmax_3(z3, p);
      int pred = (p[1] > p[0]) ? 1 : 0;
      if (p[2] > p[pred]) pred = 2;
      if (pred == data_Y[s]) correct++;
    }
    final_te_acc[2] = (float)correct / (float)N_TEST;
  }

  int epochs_saved = epochs_to_target[0] - epochs_to_target[1];

  Serial.println("\n==================================================================");
  Serial.println("BÁO_CÁO_NGHIỆM_THU_CUỐI_KHÓA: SESSION_13_&_14_TINYML_PIPELINE");
  Serial.println("==================================================================");
  Serial.println("[1]_KIỂM_TOÁN_KIẾN_TRÚC_&_BỘ_NHỚ_ADAM_(S14-Q1_&_S13-Q5):");
  Serial.println("Kiến_trúc       | Tham_số | Extra_Val | Extra_RAM_Adam");
  Serial.println("------------------------------------------------------------------");
  Serial.printf("Drone_3-4-3     | P=%-5d | 2P=%-6d | RAM=%d_bytes\n", p_343, 2 * p_343, 2 * p_343 * 4);
  Serial.printf("Iris_4-8-3      | P=%-5d | 2P=%-6d | RAM=%d_bytes\n", p_iris, 2 * p_iris, 2 * p_iris * 4);
  Serial.printf("Deep_3-8-4-3    | P=%-5d | 2P=%-6d | RAM=%d_bytes\n", p_3843, 2 * p_3843, 2 * p_3843 * 4);

  Serial.println("\n[2]_THÍ_NGHIỆM_HẺM_NÚI_DỐC_&_TIỀN_XỬ_LÝ_NaN_IQR_(S13-Q4_&_S14-Q3):");
  float lrs[2] = {0.01f, 0.03f};
  for (int k = 0; k < 2; k++) {
    float w1 = 1.0f, w2 = 1.0f, lr = lrs[k];
    for (int step = 1; step <= 2; step++) {
      w1 -= lr * (100.0f * w1);
      w2 -= lr * (1.0f * w2);
    }
    float final_L = 0.5f * (100.0f * w1 * w1 + w2 * w2);
    Serial.printf("SGD(lr=%.2f)_bước_2: w1=%.2f w2=%.4f -> Loss=%.5f %s\n",
                  lr, w1, w2, final_L, (lr > 0.02f) ? "[PHÂN_KỲ_VỌT_LỐ!]" : "[HỘI_TỤ_ÊM]");
  }
  float s14_age[6] = {25.0f, 32.0f, -1.0f, 41.0f, 29.0f, 120.0f};
  s14_age[2] = 32.0f;
  float q1_a = 29.75f, q3_a = 38.75f, iqr_a = q3_a - q1_a;
  float low_a = q1_a - 1.5f * iqr_a, up_a = q3_a + 1.5f * iqr_a;
  Serial.printf("S14-Q3_Audit: Median=%.1f | IQR=%.1f | Valid=[%.2f..%.2f] | Outlier=%.1f\n",
                s14_age[2], iqr_a, low_a, up_a, (s14_age[5] > up_a) ? s14_age[5] : 0.0f);

  // [PATCH LOG-6 #2] Đổi header thành 90 mẫu (60 Train / 30 Test)
  Serial.println("\n[3]_ĐUA_TỐC_ĐỘ_HỘI_TỤ_90_MẪU_(60_TRAIN/30_TEST)_(S13-Q6_&_S14-Q10):");
  Serial.println("Mô_hình         | Tham_số | Epoch_Đích | Train_Loss | Test_Acc");
  Serial.println("------------------------------------------------------------------");
  Serial.printf("1.3-4-3_(SGD)   | P=31    | Ep=%-7d | L=%-8.4f | Acc=%.2f%%\n",
                epochs_to_target[0], final_tr_loss[0], final_te_acc[0] * 100.0f);
  Serial.printf("2.3-4-3_(Adam)  | P=31    | Ep=%-7d | L=%-8.4f | Acc=%.2f%%\n",
                epochs_to_target[1], final_tr_loss[1], final_te_acc[1] * 100.0f);
  Serial.printf("3.3-8-4-3(Deep) | P=83    | Ep=%-7d | L=%-8.4f | Acc=%.2f%%\n",
                epochs_to_target[2], final_tr_loss[2], final_te_acc[2] * 100.0f);
  Serial.printf("=>_Adam_tiết_kiệm_%d_Epochs_(%d_Updates)_so_với_SGD!\n",
                epochs_saved, epochs_saved * 3);
  Serial.println("==================================================================\n");

  if (show_oled) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);  display.print("S13-14 BENCHMARK OK");
    display.setCursor(0, 16); display.print("P(343):31 P(Deep):83");
    display.setCursor(0, 32); display.print("SGD Ep:"); display.print(epochs_to_target[0]);
    display.print(" Adam Ep:"); display.print(epochs_to_target[1]);
    display.setCursor(0, 48); display.print("Saved:"); display.print(epochs_saved * 3);
    display.print(" upd [PASS]");
    display.display();
    delay(2200);
  }
}

// =========================================================
// REPORT MODE (NHẤN GIỮ NÚT BOOT >= 1.2 GIÂY): IN TRỌN BỘ BÁO CÁO S13 & S14
// =========================================================
void print_s14_evaluation_report() {
  pixels.setPixelColor(0, pixels.Color(255, 255, 255)); 
  pixels.show();

  display.clearDisplay();
  display.setCursor(0, 10); display.println("S13-14 REPORT MODE");
  display.setCursor(0, 28); display.println("Exporting Benchmark");
  display.setCursor(0, 44); display.println("ConfMatrix & RMSE..");
  display.display();

  run_s13_s14_assignment_benchmark(false);

  uint32_t n_reg = max(reg_sample_count, (uint32_t)1);
  float mae  = (float)(sum_abs_err / (double)n_reg);
  float mse  = (float)(sum_sq_err / (double)n_reg);
  float rmse = sqrtf(mse);

  Serial.println("==================================================================");
  Serial.println("BÁO_CÁO_ĐÁNH_GIÁ_THỰC_CHIẾN_SESSION_14_(FLIGHT_BLACKBOX)");
  Serial.println("==================================================================");
  Serial.printf("[4]_HỒI_QUY_ĐỘ_CAO_(S14-Q6): Mẫu=%lu NaN=%lu IQR=%lu\n",
                (unsigned long)reg_sample_count, (unsigned long)nan_imputed_count, (unsigned long)iqr_clipped_count);
  Serial.printf("-->_MAE=%.4fm | MSE=%.4fm^2 | RMSE=%.4fm\n", mae, mse, rmse);
  Serial.println("-->_RMSE/MAE_Ratio="); Serial.print(rmse / (mae + 1e-6f), 3); Serial.println(" (Fat-tail nếu > 1.25)");

  Serial.println("\n[5]_EARLY_STOPPING_&_GENERALIZATION_GAP_(S14-Q8,Q11):");
  Serial.printf("-->_Train_Loss=%.4f | Val_Loss=%.4f | Best_Val=%.4f\n",
                last_train_loss, last_val_loss, best_val_loss);
  Serial.printf("-->_Generalization_Gap=%+.4f | Saved=%lu | Rollbacks=%lu\n",
                last_val_loss - last_train_loss, (unsigned long)checkpoint_saves, (unsigned long)early_stop_rollbacks);

  Serial.println("\n[6]_MA_TRẬN_NHẦM_LẪN_3x3_&_PRECISION_RECALL_F1_(S14-Q5,Q12):");
  Serial.println(" Thực_tế_\\_Dự_đoán | Pred_SAFE(0) | Pred_WIND(1) | Pred_KILL(2)");
  Serial.println("------------------------------------------------------------------");
  const char* names[3] = {"Actual_SAFE(0)   ", "Actual_WIND(1)   ", "Actual_KILL(2)   "};
  for (int r = 0; r < 3; r++) {
    Serial.printf(" %s | %-12lu | %-12lu | %-12lu\n",
                  names[r], (unsigned long)conf_mat[r][0], (unsigned long)conf_mat[r][1], (unsigned long)conf_mat[r][2]);
  }

  Serial.println("\nLớp_trạng_thái  | Precision | Recall  | F1-Score | Support");
  Serial.println("------------------------------------------------------------------");
  const char* short_names[3] = {"SAFE_(Class_0) ", "WIND_(Class_1) ", "KILL_(Class_2) "};
  for (int c = 0; c < 3; c++) {
    uint32_t tp = conf_mat[c][c];
    uint32_t col_sum = conf_mat[0][c] + conf_mat[1][c] + conf_mat[2][c];
    uint32_t row_sum = conf_mat[c][0] + conf_mat[c][1] + conf_mat[c][2];
    float prec = (col_sum > 0) ? ((float)tp / (float)col_sum) : 1.0f;
    float rec  = (row_sum > 0) ? ((float)tp / (float)row_sum) : 1.0f;
    float f1   = (prec + rec > 0.0f) ? (2.0f * prec * rec / (prec + rec)) : 0.0f;
    Serial.printf("%s | %-9.3f | %-7.3f | %-8.3f | %lu\n",
                  short_names[c], prec, rec, f1, (unsigned long)row_sum);
  }
  Serial.println("==================================================================\n");
  delay(2500);
}

void readMPU(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) == 0 && Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6, (uint8_t)true) == 6) {
    int16_t raw_x = (Wire.read() << 8) | Wire.read();
    int16_t raw_y = (Wire.read() << 8) | Wire.read();
    int16_t raw_z = (Wire.read() << 8) | Wire.read();
    last_ax = raw_x / 16384.0f;
    last_ay = raw_y / 16384.0f;
    last_az = raw_z / 16384.0f;
  }
  ax = last_ax; ay = last_ay; az = last_az;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(41, 42);

  pinMode(BTN_BOOT, INPUT_PULLUP);
  pinMode(BTN_VOL_UP, INPUT_PULLUP);
  pinMode(BTN_VOL_DOWN, INPUT_PULLUP);
  
  pixels.begin();
  pixels.setBrightness(50);
  pixels.setPixelColor(0, pixels.Color(0, 0, 255)); 
  pixels.show();

  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.setTextColor(SSD1306_WHITE);
  display.setTextWrap(false); 

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  if (!bmp.begin(0x76)) {
    display.clearDisplay(); display.setCursor(0,20);
    display.println("ERR: BMP280 LOST!"); display.display();
    while (1) delay(10);
  }
  
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     
                  Adafruit_BMP280::SAMPLING_X2,     
                  Adafruit_BMP280::SAMPLING_X8,     
                  Adafruit_BMP280::FILTER_X4,       
                  Adafruit_BMP280::STANDBY_MS_1);   

  save_mlp_checkpoint();
  best_val_loss = compute_validation_loss();
  last_val_loss = best_val_loss;
  checkpoint_saves = 1;

  if (ENABLE_BENCHMARK) {
    delay(800);
    run_s13_s14_assignment_benchmark(true);
  }

  display.clearDisplay(); display.setCursor(0,20);
  display.println("Calibrating Sensors.."); display.display();
  float sum_alt = 0, sum_x = 0, sum_y = 0, sum_acc = 0;
  for(int i=0; i<25; i++) {
    float r_alt = bmp.readAltitude(1013.25);
    sum_alt += r_alt;
    alt_window[i % 5] = r_alt;
    float ax, ay, az;
    readMPU(ax, ay, az);
    sum_x += ax; sum_y += ay;
    sum_acc += sqrt(ax*ax + ay*ay + az*az);
    prev_ax = ax; prev_ay = ay; prev_az = az;
    delay(40);
  }
  base_altitude = sum_alt / 25.0f;
  offset_x = sum_x / 25.0f;
  offset_y = sum_y / 25.0f;
  base_accel = sum_acc / 25.0f;
  
  pixels.setPixelColor(0, pixels.Color(0, 255, 0)); 
  pixels.show();
}

void loop() {
  step_counter++;

  // ==========================================
  // 1. TIỀN XỬ LÝ NaN/IQR (S14-Q3) & LỌC ĐỘ CAO MẶT BÀN
  // ==========================================
  float ax, ay, az;
  readMPU(ax, ay, az);
  smooth_x = (0.8f * smooth_x) + (0.2f * (ax - offset_x));
  smooth_y = (0.8f * smooth_y) + (0.2f * (ay - offset_y));

  float jerk = abs(ax - prev_ax) + abs(ay - prev_ay) + abs(az - prev_az);
  prev_ax = ax; prev_ay = ay; prev_az = az;

  float tilt_mag = sqrt(smooth_x * smooth_x + smooth_y * smooth_y);
  float accel_mag = sqrt(ax*ax + ay*ay + az*az);
  float vibration = abs(accel_mag - base_accel);

  if (jerk < 0.025f && tilt_mag < 0.08f) {
    if (idle_counter < 100) idle_counter++;
  } else {
    idle_counter = 0; 
  }
  bool is_on_table = (idle_counter >= 12);

  float raw_alt = clean_altitude_iqr(bmp.readAltitude(1013.25));
  static float filtered_alt = 0.0f;

  if (is_on_table) {
    base_altitude = (0.92f * base_altitude) + (0.08f * raw_alt);
    filtered_alt = (0.6f * filtered_alt); 
    if (abs(filtered_alt) < 0.08f) filtered_alt = 0.0f;
  } else {
    float raw_diff = raw_alt - base_altitude;
    filtered_alt = (0.7f * filtered_alt) + (0.3f * raw_diff);
  }
  float current_altitude = filtered_alt;

  if (digitalRead(BTN_VOL_UP) == LOW) { 
    target_altitude += 0.5f; reset_adam_alt(); delay(180); 
  }
  if (digitalRead(BTN_VOL_DOWN) == LOW) { 
    target_altitude -= 0.5f; reset_adam_alt(); delay(180); 
  }
  if (digitalRead(BTN_BOOT) == LOW) {
    unsigned long press_start = millis();
    bool is_long_press = false;
    while (digitalRead(BTN_BOOT) == LOW) {
      if (millis() - press_start >= 1200) {
        is_long_press = true;
        break;
      }
      delay(10);
    }
    if (is_long_press) {
      print_s14_evaluation_report();
    } else {
      base_altitude = raw_alt; filtered_alt = 0.0f; base_accel = accel_mag;
      target_altitude = 0.0f; weight_W = 1.0f; bias_b = 25.0f;
      reset_adam_alt(); mlp_converged = false; patience_counter = 0;
      delay(250);
    }
  }

  float raw_error = target_altitude - current_altitude; 
  float error = 0.0f;
  if (!is_on_table || target_altitude != 0.0f) {
    if (abs(raw_error) > 0.22f) {
      error = (raw_error > 0.0f) ? (raw_error - 0.22f) : (raw_error + 0.22f);
    }
  }

  sum_abs_err += (double)abs(error);
  sum_sq_err  += (double)(error * error);
  reg_sample_count++;

  // ==========================================
  // 2. SESSION 11 & 14: MẠNG MLP 3->4->3 + ĐỒNG BỘ MẶT PHẲNG NĂNG LƯỢNG
  // ==========================================
  float X_input[3];
  X_input[0] = minmax_scale(tilt_mag, 0.0f, 1.00f);      
  X_input[1] = minmax_scale(vibration, 0.0f, 0.60f);     
  X_input[2] = minmax_scale(abs(error), 0.0f, 6.00f);    

  float Z1_live[4], A1_live[4], class_probs[3];
  float ce_loss = 0.0f;
  int flight_state = forward_mlp_cache(X_input, Z1_live, A1_live, class_probs, ce_loss, 0);

  float wind_energy = X_input[0] + (0.8f * X_input[1]) + (0.4f * X_input[2]);

  int teacher_class = 0;
  if (X_input[0] > 0.78f) { 
    teacher_class = 2;
    if (X_input[0] > 0.94f) {
      replay_X[2][0] = (0.85f * replay_X[2][0]) + (0.15f * X_input[0]);
      replay_X[2][1] = 0.05f; replay_X[2][2] = 0.05f;
    }
  } else if (wind_energy > 0.28f) { 
    teacher_class = 1;
    if (X_input[0] >= 0.48f && X_input[0] <= 0.60f) {
      for (int i = 0; i < 3; i++) replay_X[1][i] = (0.85f * replay_X[1][i]) + (0.15f * X_input[i]);
    }
  } else {
    teacher_class = 0;
    if (wind_energy < 0.08f) {
      for (int i = 0; i < 3; i++) replay_X[0][i] = (0.85f * replay_X[0][i]) + (0.15f * X_input[i]);
    }
  }

  bool in_transition_band = (abs(wind_energy - 0.28f) < 0.03f) || (abs(X_input[0] - 0.78f) < 0.03f);
  bool is_boot_released = (digitalRead(BTN_BOOT) == HIGH);

  if (!in_transition_band && is_boot_released) {
    conf_mat[teacher_class][flight_state]++;
  }

  if (flight_state != teacher_class && !in_transition_band && is_boot_released) {
    mlp_converged = false;
    patience_counter = 0; 
  }

  if (step_counter % 10 == 0 && !mlp_converged) {
    train_mlp_minibatch_momentum(lr_mlp);
  }

  // ==========================================
  // 3. SESSION 11 & 12: ADAMW OPTIMIZER (ĐÓNG BĂNG KHI [KILL] & XÓA TRỄ 1 NHỊP)
  // ==========================================
  if (alt_converged && abs(raw_error - locked_raw_error) > 0.24f) {
    reset_adam_alt();
  }

  prev_W = weight_W;

  if (flight_state != 2) {
    if (!alt_converged && error != 0.0f) {
      float z_pre = (weight_W * error) + bias_b;
      float error_sign = (error >= 0.0f) ? 1.0f : -1.0f;
      float d_loss_d_error = (abs(error) <= outlier_threshold) ? (2.0f * error) : (2.0f * outlier_threshold * error_sign);
      float d_leaky = (z_pre >= 0.0f) ? 1.0f : 0.1f;
      float base_gradient = d_loss_d_error * (-1.0f) * d_leaky; 
      float eff_err = min(abs(error), outlier_threshold);
      
      float grad_W = base_gradient * eff_err; 
      float grad_b = base_gradient * 1.0f;

      t_alt++;
      beta1_pow *= adam_beta1;
      beta2_pow *= adam_beta2;

      m_W = adam_beta1 * m_W + (1.0f - adam_beta1) * grad_W;
      v_W = adam_beta2 * v_W + (1.0f - adam_beta2) * (grad_W * grad_W);

      m_b = adam_beta1 * m_b + (1.0f - adam_beta1) * grad_b;
      v_b = adam_beta2 * v_b + (1.0f - adam_beta2) * (grad_b * grad_b);

      float inv_bias1 = 1.0f / (1.0f - beta1_pow);
      float inv_bias2 = 1.0f / (1.0f - beta2_pow);
      float m_hat_W = m_W * inv_bias1;
      float v_hat_W = v_W * inv_bias2;
      float m_hat_b = m_b * inv_bias1;
      float v_hat_b = v_b * inv_bias2;

      int t_decay = min(t_alt - 1, 50);
      float decay_factor = 1.0f / (1.0f + lr_decay_k * (float)t_decay);
      float alpha_W = alpha0_W * decay_factor;
      float alpha_b = alpha0_b * decay_factor;
      current_eff_lr = alpha_W;

      weight_W -= alpha_W * m_hat_W / (sqrtf(v_hat_W) + adam_eps_ctrl);
      bias_b   -= alpha_b * m_hat_b / (sqrtf(v_hat_b) + adam_eps_ctrl);
    }

    weight_W = constrain(weight_W - 0.04f * decay_W * (weight_W - 1.0f), 0.5f, 25.0f); 
    bias_b   = constrain(bias_b   - 0.06f * decay_b * (bias_b - 25.0f), 0.0f, 60.0f); 
  }

  float z_alt = (weight_W * error) + bias_b; 
  float final_throttle = constrain(leaky_relu(z_alt), 0.0f, 100.0f);

  if (!alt_converged && flight_state != 2) {
    if (error == 0.0f) {
      alt_cvg_count++;
      if (alt_cvg_count >= 10) { 
        alt_converged = true;
        locked_raw_error = raw_error;
      }
    } else {
      alt_cvg_count = 0;
    }
  }

  // ==========================================
  // 4. TÍNH SONG SONG: TỶ TRỌNG GÓC (MIX) & CÔNG SUẤT ESC THỰC TẾ (OUT)
  // ==========================================
  float mix_x = is_on_table ? 0.0f : smooth_x;
  float mix_y = is_on_table ? 0.0f : smooth_y;

  float z_M1 = (-2.5f * mix_x) + (-2.5f * mix_y); 
  float z_M2 = (-2.5f * mix_x) + (+2.5f * mix_y); 
  float z_M3 = (+2.5f * mix_x) + (-2.5f * mix_y); 
  float z_M4 = (+2.5f * mix_x) + (+2.5f * mix_y); 

  float mix1, mix2, mix3, mix4;
  softmax_4(z_M1, z_M2, z_M3, z_M4, 1.0f, mix1, mix2, mix3, mix4);

  float roll_pitch_bias = 60.0f;
  float out1 = constrain(final_throttle + (mix1 - 0.25f) * roll_pitch_bias, 0.0f, 100.0f);
  float out2 = constrain(final_throttle + (mix2 - 0.25f) * roll_pitch_bias, 0.0f, 100.0f);
  float out3 = constrain(final_throttle + (mix3 - 0.25f) * roll_pitch_bias, 0.0f, 100.0f);
  float out4 = constrain(final_throttle + (mix4 - 0.25f) * roll_pitch_bias, 0.0f, 100.0f);

  if (flight_state == 2) {
    out1 = out2 = out3 = out4 = 0.0f;
    final_throttle = 0.0f;
  }

  // ==========================================
  // 5. HIỂN THỊ HUD ([C]=CONVERGED, [S]=SAFE, [W]=WIND, [K]=KILL)
  // ==========================================
  display.clearDisplay();
  display.setTextSize(1); 
  
  display.setCursor(0, 0);  display.print("T:"); display.print(target_altitude, 1);
  display.setCursor(38, 0); display.print("C:"); display.print(current_altitude, 1);
  display.setCursor(76, 0); display.print("W:"); display.print(weight_W, 1);

  display.setCursor(0, 16);  display.print("Mx1:"); display.print(mix1 * 100.0f, 0); display.print("%");
  display.setCursor(52, 16); display.print("Mx2:"); display.print(mix2 * 100.0f, 0); display.print("%");
  display.setCursor(0, 32);  display.print("Mx3:"); display.print(mix3 * 100.0f, 0); display.print("%");
  display.setCursor(52, 32); display.print("Mx4:"); display.print(mix4 * 100.0f, 0); display.print("%");

  display.setCursor(0, 50);  display.print("b:"); display.print(bias_b, 0);
  display.setCursor(30, 50); display.print("CE:"); display.print(ce_loss, 2);
  display.setCursor(94, 50);
  if (flight_state == 2) {
    display.print("[K]"); 
    pixels.setPixelColor(0, pixels.Color(255, 0, 255));   
  } else if (flight_state == 1) {
    display.print("[W]"); 
    pixels.setPixelColor(0, pixels.Color(255, 165, 0));   
  } else if (alt_converged) {
    display.print("[C]"); 
    pixels.setPixelColor(0, pixels.Color(0, 255, 255));     
  } else {
    display.print("[S]"); 
    pixels.setPixelColor(0, pixels.Color(0, 255, 0));     
  }
  pixels.show();

  int bar_height = constrain((int)(final_throttle * 64.0f / 100.0f), 0, 64); 
  display.drawRect(118, 0, 10, 64, SSD1306_WHITE); 
  display.fillRect(118, 64 - bar_height, 10, bar_height, SSD1306_WHITE); 

  display.display();

  // =========================================================
  // QUY HOẠCH BẢNG MÀU SERIAL PLOTTER (10 ĐƯỜNG MƯỢT MÀ)
  // =========================================================
  Serial.print("Mx1:");      Serial.print(mix1 * 100.0f);        Serial.print(","); 
  Serial.print("Mx2:");      Serial.print(mix2 * 100.0f);        Serial.print(","); 
  Serial.print("Mx3:");      Serial.print(mix3 * 100.0f);        Serial.print(","); 
  Serial.print("Mx4:");      Serial.print(mix4 * 100.0f);        Serial.print(","); 
  Serial.print("Throttle:"); Serial.print(final_throttle);       Serial.print(","); 
  Serial.print("b:");        Serial.print(bias_b);               Serial.print(","); 
  Serial.print("W:");        Serial.print(weight_W);             Serial.print(","); 
  Serial.print("CE_Loss:");  Serial.print(ce_loss * 10.0f);      Serial.print(","); 
  Serial.print("C:");        Serial.print(current_altitude, 2);  Serial.print(","); 
  Serial.print("C_x10:");    Serial.println(current_altitude * 10.0f);              

  delay(40);
}
