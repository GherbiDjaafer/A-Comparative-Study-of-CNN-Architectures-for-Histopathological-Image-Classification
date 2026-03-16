# ============================================================================
# 1. IMPORTS
# ============================================================================

# Standard library
import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

# Data processing and scientific computing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    matthews_corrcoef, cohen_kappa_score
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300

# Deep Learning
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Conv2D, Layer,
    Add, Activation, Multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import mixed_precision

# Google Colab
from google.colab import drive

print("="*80)
print("SYSTEM INFORMATION")
print("="*80)
print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"✓ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("✓ All imports successful\n")

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for reproducibility"""
    
    # Experiment metadata
    EXPERIMENT_NAME = f"EfficientNetB0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Random seeds for reproducibility
    SEED = 42
    
    # Paths
    DATA_PATH = '/kaggle/input/kmc-renal/KMC_Dataset_for_grading (1)'
    OUTPUT_PATH = '/kaggle/working'
    
    # Dataset parameters
    CATEGORIES = ['0 (1)', '1 (1)', '2 (1)', '3 (1)', '4 (1)']
    NUM_CLASSES = len(CATEGORIES)
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 50
    IMG_SIZE = (224, 224)        
    BATCH_SIZE = 8               
    INITIAL_LR = 5e-4
    FINE_TUNE_LR = 5e-5
    
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.2
        
        
    # Model architecture
    CBAM_RATIO = 8
    CBAM_KERNEL = 7
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.2
    total_layers = len(EfficientNetB0(include_top=False).layers)
    UNFREEZE_FROM_LAYER = int(total_layers * 0.70)
        
    # Data split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Hardware optimization
    USE_MIXED_PRECISION = True
    
    @classmethod
    def get_output_dir(cls):
        """Create and return experiment-specific output directory"""
        exp_dir = os.path.join(cls.OUTPUT_PATH, cls.EXPERIMENT_NAME)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
        return exp_dir

# Set random seeds
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)

print("="*80)
print("CONFIGURATION")
print("="*80)
print(f"Experiment: {Config.EXPERIMENT_NAME}")
print(f"Image Size: {Config.IMG_SIZE}")
print(f"Batch Size: {Config.BATCH_SIZE}")
print(f"Classes: {Config.NUM_CLASSES}")
print(f"Mixed Precision: {Config.USE_MIXED_PRECISION}\n")

# ============================================================================
# 3. SETUP FUNCTIONS
# ============================================================================

def setup_environment():
    """Setup Google Drive and GPU"""
    print("="*80)
    print("ENVIRONMENT SETUP")
    print("="*80)
    
    # Setup GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Using {len(gpus)} GPU(s)")
            
            if Config.USE_MIXED_PRECISION:
                mixed_precision.set_global_policy('mixed_float16')
                print("✓ Mixed precision enabled (float16)")
        except RuntimeError as e:
            print(f"✗ GPU setup error: {e}")
    else:
        print("⚠ No GPUs detected, using CPU")
    
    # Create output directories
    output_dir = Config.get_output_dir()
    print(f"✓ Output directory: {output_dir}\n")
    
    return output_dir

# ============================================================================
# 4. DATA LOADING AND PREPARATION
# ============================================================================

def load_dataset():
    """Load images and labels from directory structure"""
    print("="*80)
    print("DATA LOADING")
    print("="*80)
    
    image_paths = []
    labels = []
    
    for category in Config.CATEGORIES:
        category_path = os.path.join(Config.DATA_PATH, category)
        
        if not os.path.exists(category_path):
            print(f"⚠ Warning: {category_path} not found")
            continue
        
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        for img in images:
            image_paths.append(os.path.join(category_path, img))
            labels.append(category)
    
    df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels
    })
    
    # Encode labels
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["label"])
    
    print(f"✓ Total images loaded: {len(df)}")
    print(f"✓ Classes: {Config.CATEGORIES}")
    print("\nClass distribution:")
    print(df["label"].value_counts().sort_index())
    print()
    
    return df, le

def split_dataset(df):
    """Stratified train/val/test split"""
    print("="*80)
    print("DATA SPLITTING")
    print("="*80)
    
    # Train: 80%, Temp: 20%
    train_df, temp_df = train_test_split(
        df, train_size=Config.TRAIN_RATIO,
        random_state=Config.SEED,
        stratify=df["category_encoded"]
    )
    
    # Val: 10%, Test: 10%
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        random_state=Config.SEED,
        stratify=temp_df["category_encoded"]
    )
    
    print(f"✓ Training:   {len(train_df):5d} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✓ Validation: {len(valid_df):5d} samples ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"✓ Test:       {len(test_df):5d} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\nClass distribution per split:")
    for split_name, split_df in [("Train", train_df), ("Val", valid_df), ("Test", test_df)]:
        print(f"\n{split_name}:")
        print(split_df["label"].value_counts().sort_index())
    print()
    
    return train_df, valid_df, test_df

def balance_training_data(train_df):
    """Oversample minority classes in training set"""
    print("="*80)
    print("DATA BALANCING (Training Set Only)")
    print("="*80)
    
    print("Original distribution:")
    print(train_df["category_encoded"].value_counts().sort_index())
    
    max_count = train_df["category_encoded"].value_counts().max()
    train_balanced = []
    
    for cls in sorted(train_df["category_encoded"].unique()):
        cls_df = train_df[train_df["category_encoded"] == cls]
        cls_upsampled = resample(
            cls_df, replace=True,
            n_samples=max_count,
            random_state=Config.SEED
        )
        train_balanced.append(cls_upsampled)
    
    train_df_balanced = pd.concat(train_balanced).sample(frac=1, random_state=Config.SEED)
    
    print("\nBalanced distribution:")
    print(train_df_balanced["category_encoded"].value_counts().sort_index())
    print(f"✓ Total training samples after balancing: {len(train_df_balanced)}\n")
    
    return train_df_balanced

# ============================================================================
# 5. DATA AUGMENTATION
# ============================================================================

def create_data_generators(train_df, valid_df, test_df):
    """Create data generators with augmentation"""
    print("="*80)
    print("DATA AUGMENTATION")
    print("="*80)


    
    # Training augmentation (histopathology-appropriate)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.95, 1.05],
        fill_mode='reflect'
    )
    
    # Validation/Test (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="image_path", y_col="category_encoded",
        target_size=Config.IMG_SIZE, class_mode="raw",
        batch_size=Config.BATCH_SIZE, shuffle=True, seed=Config.SEED
    )
    
    valid_gen = val_test_datagen.flow_from_dataframe(
        valid_df, x_col="image_path", y_col="category_encoded",
        target_size=Config.IMG_SIZE, class_mode="raw",
        batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, x_col="image_path", y_col="category_encoded",
        target_size=Config.IMG_SIZE, class_mode="raw",
        batch_size=Config.BATCH_SIZE, shuffle=False
    )
    
    print(f"✓ Training batches:   {len(train_gen)}")
    print(f"✓ Validation batches: {len(valid_gen)}")
    print(f"✓ Test batches:       {len(test_gen)}\n")
    
    return train_gen, valid_gen, test_gen

def unfreeze_top_percentage(base_model, freeze_ratio=0.7):
    """
    Freeze bottom `freeze_ratio` of layers and unfreeze the rest.
    Example: freeze_ratio=0.7 → unfreeze top 30%
    """
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * freeze_ratio)

    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    print(f"✓ Total layers in base model: {total_layers}")
    print(f"✓ Frozen layers: {unfreeze_from}")
    print(f"✓ Trainable layers: {total_layers - unfreeze_from}")

    return unfreeze_from


def CBAMLayer(x, ratio=8):
    filters = x.shape[-1]
    from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Conv2D, Layer,
    Add, Activation, Multiply,GlobalMaxPooling2D,Reshape, Lambda, Concatenate
)
    # CHANNEL ATTENTION
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    
    avg_pool = Reshape((1, 1, filters))(avg_pool)
    max_pool = Reshape((1, 1, filters))(max_pool)
    
    # Shared MLP
    dense1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal')
    dense2 = Dense(filters, kernel_initializer='he_normal')
    
    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))
    
    channel_attention = Activation('sigmoid')(Add()([avg_out, max_out]))
    x = Multiply()([x, channel_attention])
    
    # SPATIAL ATTENTION - Using Lambda layers for TF operations
    avg_pool_spatial = Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool_spatial = Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
    
    spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid',
                               kernel_initializer='he_normal')(concat)
    x = Multiply()([x, spatial_attention])
    
    return x



# ============================================================================
# 7. MODEL ARCHITECTURE
# ============================================================================

def build_model():
    print("="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)

    base_model = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(*Config.IMG_SIZE, 3))
    base_model.trainable = False

    x = base_model.output
    #x = CBAMLayer(x=x,ratio=Config.CBAM_RATIO)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(Config.DROPOUT_1, name='dropout1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(Config.DROPOUT_2, name='dropout2')(x)

    predictions = Dense(
        Config.NUM_CLASSES,
        activation='softmax',
        dtype='float32',
        name='output'
    )(x)

    model = Model(inputs=base_model.input, outputs=predictions,
                  name='EfficientNetB0')

    print(f"✓ Model: {model.name}")
    print(f"✓ Total parameters: {model.count_params():,}\n")

    return model, base_model

# ============================================================================
# 8. TRAINING
# ============================================================================

def train_model(model, base_model, train_gen, valid_gen, output_dir):
    """Two-phase training: freeze then fine-tune"""
    
    # ========== PHASE 1: FREEZE BASE ==========
    print("="*80)
    print("PHASE 1: INITIAL TRAINING (FROZEN BASE)")
    print("="*80)
    
    model.compile(
        optimizer=Adam(learning_rate=Config.INITIAL_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    callbacks_phase1 = [
        ModelCheckpoint(
            os.path.join(output_dir, 'models', 'best_model_phase1.keras'),
            save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1
        ),
        CSVLogger(os.path.join(output_dir, 'logs', 'training_phase1.csv')),
        TensorBoard(log_dir=os.path.join(output_dir, 'logs', 'tensorboard_phase1'))
    ]
    
    print(f"Learning Rate: {Config.INITIAL_LR}")
    print(f"Epochs: {Config.INITIAL_EPOCHS}\n")
    
    history_phase1 = model.fit(
        train_gen, validation_data=valid_gen,
        epochs=Config.INITIAL_EPOCHS,
        callbacks=callbacks_phase1, verbose=1
    )
    
    # ========== PHASE 2: FINE-TUNING ==========
    print("\n" + "="*80)
    print("PHASE 2: FINE-TUNING (UNFROZEN LAYERS)")
    print("="*80)
    
    # Unfreeze top layers
    unfreeze_from = unfreeze_top_percentage(base_model, freeze_ratio=0.7)
    trainable_layers = sum([1 for l in model.layers if l.trainable])
    frozen_layers = sum([1 for l in model.layers if not l.trainable])
    
    print(f"✓ Trainable layers: {trainable_layers}")
    print(f"✓ Frozen layers:    {frozen_layers}")
    
    model.compile(
        optimizer=Adam(learning_rate=Config.FINE_TUNE_LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    callbacks_phase2 = [
        ModelCheckpoint(
            os.path.join(output_dir, 'models', 'best_model_final.keras'),
            save_best_only=True, monitor='val_accuracy', mode='max', verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-7, verbose=1
        ),
        CSVLogger(os.path.join(output_dir, 'logs', 'training_phase2.csv')),
        TensorBoard(log_dir=os.path.join(output_dir, 'logs', 'tensorboard_phase2'))
    ]
    
    total_epochs = Config.INITIAL_EPOCHS + Config.FINE_TUNE_EPOCHS
    
    print(f"Learning Rate: {Config.FINE_TUNE_LR}")
    print(f"Total Epochs: {total_epochs}\n")
    
    history_phase2 = model.fit(
        train_gen, validation_data=valid_gen,
        epochs=total_epochs,
        initial_epoch=history_phase1.epoch[-1] + 1,
        callbacks=callbacks_phase2, verbose=1
    )
    
    print("\n✓ Training complete\n")
    return history_phase1, history_phase2

# ============================================================================
# 9. COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_model(model, test_gen, class_names, output_dir):
    """Comprehensive evaluation with research metrics"""
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get predictions
    test_gen.reset()
    y_true = []
    y_pred_probs = []
    
    print("Generating predictions...")
    for i in range(len(test_gen)):
        images, labels = next(test_gen)
        preds = model.predict(images, verbose=0)
        y_true.extend(labels)
        y_pred_probs.extend(preds)
    
    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # ========== BASIC METRICS ==========
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)
    
    print("\n" + "-"*80)
    print("OVERALL METRICS")
    print("-"*80)
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss:      {test_loss:.4f}")
    
    # ========== ADVANCED METRICS ==========
    # Weighted F1
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"Weighted F1:    {weighted_f1:.4f}")
    print(f"Macro F1:       {macro_f1:.4f}")
    print(f"MCC:            {mcc:.4f}")
    print(f"Cohen's Kappa:  {kappa:.4f}")
    
    # ========== PER-CLASS METRICS ==========
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        digits=4, output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # ========== CONFUSION MATRIX ==========
    cm = confusion_matrix(y_true, y_pred)
    
    # ========== ROC-AUC & PR-AUC ==========
    print("-"*80)
    print("ROC-AUC & PR-AUC SCORES")
    print("-"*80)
    
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    roc_auc_scores = {}
    pr_auc_scores = {}
    
    for i, class_name in enumerate(class_names):
        if len(np.unique(y_true_bin[:, i])) > 1:
            # ROC-AUC
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc_scores[class_name] = auc(fpr, tpr)
            
            # PR-AUC
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
            pr_auc_scores[class_name] = auc(recall, precision)
            
            print(f"{class_name:10s} - ROC-AUC: {roc_auc_scores[class_name]:.4f}, PR-AUC: {pr_auc_scores[class_name]:.4f}")
    
    # Macro average
    macro_roc_auc = np.mean(list(roc_auc_scores.values()))
    macro_pr_auc = np.mean(list(pr_auc_scores.values()))
    print(f"\nMacro Average - ROC-AUC: {macro_roc_auc:.4f}, PR-AUC: {macro_pr_auc:.4f}")
    
    # ========== SAVE RESULTS ==========
    results = {
        'experiment_name': Config.EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'weighted_f1': float(weighted_f1),
            'macro_f1': float(macro_f1),
            'mcc': float(mcc),
            'cohens_kappa': float(kappa),
            'macro_roc_auc': float(macro_roc_auc),
            'macro_pr_auc': float(macro_pr_auc)
        },
        'per_class_metrics': {
            'roc_auc': roc_auc_scores,
            'pr_auc': pr_auc_scores,
            'classification_report': report
        },
        'predictions': {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_probs': y_pred_probs.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'configuration': {
            'seed': Config.SEED,
            'image_size': Config.IMG_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'initial_lr': Config.INITIAL_LR,
            'finetune_lr': Config.FINE_TUNE_LR
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved to: {os.path.join(output_dir, 'evaluation_results.json')}\n")
    
    return results, y_true, y_pred_probs

# ============================================================================
# 10. VISUALIZATION
# ============================================================================

def plot_training_history(history1, history2, output_dir):
    """Plot comprehensive training history"""
    print("Generating training history plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Combine histories
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    ft_start = len(history1.history['accuracy'])
    
    # Plot 1: Accuracy
    axes[0, 0].plot(epochs, acc, 'b-o', label='Training', linewidth=2, markersize=4)
    axes[0, 0].plot(epochs, val_acc, 'r-s', label='Validation', linewidth=2, markersize=4)
    axes[0, 0].axvline(ft_start, color='green', linestyle='--', linewidth=2, label='Fine-tuning starts')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss
    axes[0, 1].plot(epochs, loss, 'b-o', label='Training', linewidth=2, markersize=4)
    axes[0, 1].plot(epochs, val_loss, 'r-s', label='Validation', linewidth=2, markersize=4)
    axes[0, 1].axvline(ft_start, color='green', linestyle='--', linewidth=2, label='Fine-tuning starts')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Generalization Gap
    acc_diff = np.array(acc) - np.array(val_acc)
    axes[1, 0].plot(epochs, acc_diff, color='purple', linewidth=2)
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(ft_start, color='green', linestyle='--', linewidth=2)
    axes[1, 0].fill_between(epochs, 0, acc_diff, alpha=0.3, color='purple')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Train Acc - Val Acc', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Generalization Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy Trend
    val_acc_smooth = pd.Series(val_acc).rolling(window=3, min_periods=1).mean()
    axes[1, 1].plot(epochs, val_acc, 'o-', alpha=0.4, label='Raw', markersize=4)
    axes[1, 1].plot(epochs, val_acc_smooth, 'r-', linewidth=3, label='Smoothed (MA-3)')
    axes[1, 1].axvline(ft_start, color='green', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Validation Accuracy Trend', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'fig1_training_history.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig1_training_history.png")

def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot normalized and raw confusion matrices"""
    print("Generating confusion matrix plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Proportion'},
                linewidths=0.5, linecolor='gray')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Raw counts confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Count'},
                linewidths=0.5, linecolor='gray')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'fig2_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig2_confusion_matrix.png")

def plot_roc_curves(y_true, y_pred_probs, class_names, output_dir):
    """Plot ROC curves for all classes"""
    print("Generating ROC curves...")
    
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5, 
                   label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'fig3_roc_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig3_roc_curves.png")

def plot_precision_recall_curves(y_true, y_pred_probs, class_names, output_dir):
    """Plot Precision-Recall curves"""
    print("Generating Precision-Recall curves...")
    
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    for i, (name, color) in enumerate(zip(class_names, colors)):
        if len(np.unique(y_true_bin[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, color=color, lw=2.5,
                   label=f'{name} (AUC = {pr_auc:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'fig4_precision_recall_curves.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig4_precision_recall_curves.png")

def plot_per_class_metrics(report, class_names, output_dir):
    """Plot per-class performance metrics"""
    print("Generating per-class metrics plot...")
    
    metrics = ['precision', 'recall', 'f1-score']
    data = {m: [report[name][m] for name in class_names] for m in metrics}
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, data['precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, data['recall'], width, label='Recall', color='#e74c3c')
    bars3 = ax.bar(x + width, data['f1-score'], width, label='F1-Score', color='#2ecc71')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'fig5_per_class_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig5_per_class_metrics.png")

def plot_class_distribution(train_df, valid_df, test_df, output_dir):
    """Plot class distribution across splits"""
    print("Generating class distribution plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for ax, (name, df) in zip(axes, [('Training', train_df), 
                                      ('Validation', valid_df), 
                                      ('Test', test_df)]):
        counts = df['label'].value_counts().sort_index()
        ax.bar(range(len(counts)), counts.values, color='steelblue', edgecolor='black')
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_xlabel('Class', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{name} Set (n={len(df)})', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        for i, v in enumerate(counts.values):
            ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'fig6_class_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fig6_class_distribution.png")

# ============================================================================
# 11. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("EfficientNetB0 TRAINING PIPELINE")
    print("="*80 + "\n")
    
    # 1. Setup
    output_dir = setup_environment()
    
    # 2. Load data
    df, label_encoder = load_dataset()
    
    # 3. Split data
    train_df, valid_df, test_df = split_dataset(df)
    
    # 4. Balance training data
    train_df_balanced = balance_training_data(train_df)
    
    # 5. Create data generators
    train_gen, valid_gen, test_gen = create_data_generators(
        train_df_balanced, valid_df, test_df
    )
    
    # 6. Plot initial data distribution
    plot_class_distribution(train_df, valid_df, test_df, output_dir)
    
    # 7. Build model
    model, base_model = build_model()
    
    # 8. Train model
    history1, history2 = train_model(model, base_model, train_gen, valid_gen, output_dir)
    
    # 9. Plot training history
    plot_training_history(history1, history2, output_dir)
    
    # 10. Evaluate model
    results, y_true, y_pred_probs = evaluate_model(
        model, test_gen, Config.CATEGORIES, output_dir
    )
    
    # 11. Generate all visualizations
    cm = confusion_matrix(y_true, np.argmax(y_pred_probs, axis=1))
    plot_confusion_matrix(cm, Config.CATEGORIES, output_dir)
    plot_roc_curves(y_true, y_pred_probs, Config.CATEGORIES, output_dir)
    plot_precision_recall_curves(y_true, y_pred_probs, Config.CATEGORIES, output_dir)
    plot_per_class_metrics(results['per_class_metrics']['classification_report'], 
                          Config.CATEGORIES, output_dir)
    
    # 12. Save final model
    final_model_path = os.path.join(output_dir, 'models', 'final_model.keras')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # 13. Save experiment summary
    summary = {
        'experiment_name': Config.EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'configuration': vars(Config),
        'data_split': {
            'train': len(train_df),
            'validation': len(valid_df),
            'test': len(test_df)
        },
        'final_metrics': results['overall_metrics'],
        'output_directory': output_dir
    }
    
    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nFinal Test Accuracy: {results['overall_metrics']['test_accuracy']:.4f}")
    print(f"Macro F1-Score: {results['overall_metrics']['macro_f1']:.4f}")
    print(f"Cohen's Kappa: {results['overall_metrics']['cohens_kappa']:.4f}")
    print("\n" + "="*80 + "\n")

# ============================================================================
# EXECUTE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()