from keras.applications import (
    VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, 
    DenseNet201, DenseNet169, DenseNet121,
    InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2,
    MobileNetV3Small, MobileNetV3Large, NASNetMobile, NASNetLarge,
    EfficientNetB0, EfficientNetB1, EfficientNetB2,
    Xception
)

# 모든 모델 공통 input shape (RGB)
input_shape = (224, 224, 3)

model_list = [
    VGG16(weights='imagenet', include_top=False, input_shape=input_shape),
    VGG19(weights='imagenet', include_top=False, input_shape=input_shape),

    ResNet50(weights='imagenet', include_top=False, input_shape=input_shape),
    ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape),
    ResNet101(weights='imagenet', include_top=False, input_shape=input_shape),
    ResNet101V2(weights='imagenet', include_top=False, input_shape=input_shape),
    ResNet152(weights='imagenet', include_top=False, input_shape=input_shape),
    ResNet152V2(weights='imagenet', include_top=False, input_shape=input_shape),

    DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape),
    DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape),
    DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape),

    InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape),
    InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape),

    MobileNet(weights='imagenet', include_top=False, input_shape=input_shape),
    MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
    MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape),
    MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape),

    NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape),
    NASNetLarge(weights='imagenet', include_top=False, input_shape=input_shape),

    # EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape),
    # EfficientNetB1(weights='imagenet', include_top=False, input_shape=input_shape),
    # EfficientNetB2(weights='imagenet', include_top=False, input_shape=input_shape),

    Xception(weights='imagenet', include_top=False, input_shape=input_shape)
]

# 모델별 가중치 상태 출력
for model in model_list:
    model.trainable = False

    print("=" * 100)
    print("모델명:", model.name)
    print("전체 가중치 갯수:", len(model.weights))
    print("훈련 가능 갯수:", len(model.trainable_weights))