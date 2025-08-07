import tensorflow as tf
from keras.applications import (
    VGG16, ResNet50, ResNet101, DenseNet121, MobileNetV2, NASNetMobile
)
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import numpy as np
import time

# CIFAR-10 준비
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
input_shape = (32, 32, 3)
num_classes = 10

# 사용할 모델 리스트
model_list = [
    VGG16(include_top=False, input_shape=input_shape),
    ResNet50(include_top=False, input_shape=input_shape),
    ResNet101(include_top=False, input_shape=input_shape),
    DenseNet121(include_top=False, input_shape=input_shape),
    MobileNetV2(include_top=False, input_shape=input_shape),
    NASNetMobile(include_top=False, input_shape=input_shape),
]

# 결과 저장 리스트
results = []

# 모델별 학습 및 평가
for base_model in model_list:
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"\n🟩 모델 학습 시작: {base_model.name}")

    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=20,              # 속도 빠르게 비교용 (실제 비교 시 20 이상 권장)
        batch_size=64,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        verbose=0
    )
    end_time = time.time()

    # 최종 평가
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    results.append({
        "model": base_model.name,
        "trainable_params": sum([np.prod(w.shape) for w in model.trainable_weights]),
        "val_acc": round(max(history.history["val_accuracy"]), 4),
        "val_loss": round(min(history.history["val_loss"]), 4),
        "test_acc": round(acc, 4),
        "test_loss": round(loss, 4),
        "time_sec": round(end_time - start_time, 1)
    })

# 결과 정렬 및 출력
results.sort(key=lambda x: x["test_acc"], reverse=True)

print("\n최종 성능 비교")
print("=" * 100)
print(f"{'모델':<20} {'파라미터':<15} {'ValAcc':<10} {'TestAcc':<10} {'Time(s)':<10}")
print("-" * 100)
for r in results:
    print(f"{r['model']:<20} {r['trainable_params']:<15} {r['val_acc']:<10} {r['test_acc']:<10} {r['time_sec']:<10}")
print("=" * 100)


# ====================================================================================================
# 모델                   파라미터            ValAcc     TestAcc    Time(s)   
# ----------------------------------------------------------------------------------------------------
# densenet121          10250           0.5622     0.5729     146.4     
# vgg16                5130            0.4778     0.4661     61.7      
# nasnet_mobile        10570           0.4032     0.3975     225.4     
# resnet50             20490           0.2962     0.2991     102.7     
# mobilenetv2_1.00_224 12810           0.2837     0.2791     80.1      
# resnet101            20490           0.2097     0.2034     145.3     
# ====================================================================================================