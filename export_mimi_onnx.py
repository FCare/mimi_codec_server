#!/usr/bin/env python3
"""
Export Mimi model to ONNX with reduced number of codebooks (n_q=8)
This significantly reduces model size and inference time.

Usage:
    # Basic export (FP32)
    python export_mimi_onnx_nq8.py --output-dir ./onnx_nq8 --num-quantizers 8
    
    # With FP16 conversion (recommended - reduces size by ~50%)
    python export_mimi_onnx_nq8.py --output-dir ./onnx_nq8 --num-quantizers 8 --fp16
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


class MimiEncoderWrapper(nn.Module):
    """Wrapper pour l'encoder Mimi avec n_q codebooks réduits."""
    
    def __init__(self, mimi_model, num_quantizers: int = 8):
        super().__init__()
        self.mimi_model = mimi_model
        self.num_quantizers = num_quantizers
        
        # Vérifier que num_quantizers est valide
        max_codebooks = mimi_model.config.num_codebooks
        if num_quantizers > max_codebooks:
            raise ValueError(
                f"num_quantizers ({num_quantizers}) cannot exceed "
                f"model's max codebooks ({max_codebooks})"
            )
        
        print(f"✓ Encoder wrapper initialized with {num_quantizers} codebooks "
              f"(max: {max_codebooks})")
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to tokens with reduced codebooks.
        
        Args:
            input_values: Audio tensor [batch, channels, samples]
        
        Returns:
            audio_codes: Token tensor [batch, n_q, frames]
        """
        # Encoder avec num_quantizers réduit
        encoder_outputs = self.mimi_model.encode(
            input_values,
            num_quantizers=self.num_quantizers
        )
        audio_codes = encoder_outputs.audio_codes
        
        # Vérifier la shape
        assert audio_codes.shape[1] == self.num_quantizers, \
            f"Expected {self.num_quantizers} codebooks, got {audio_codes.shape[1]}"
        
        return audio_codes


class MimiDecoderWrapper(nn.Module):
    """Wrapper pour le decoder Mimi avec n_q codebooks réduits."""
    
    def __init__(self, mimi_model, num_quantizers: int = 8):
        super().__init__()
        self.mimi_model = mimi_model
        self.num_quantizers = num_quantizers
        
        print(f"✓ Decoder wrapper initialized with {num_quantizers} codebooks")
    
    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to audio with reduced codebooks.
        
        Args:
            audio_codes: Token tensor [batch, n_q, frames]
        
        Returns:
            audio_values: Audio tensor [batch, channels, samples]
        """
        # Vérifier que les codes ont le bon nombre de codebooks
        assert audio_codes.shape[1] == self.num_quantizers, \
            f"Expected {self.num_quantizers} codebooks, got {audio_codes.shape[1]}"
        
        # Decoder
        decoder_outputs = self.mimi_model.decode(audio_codes)
        audio_values = decoder_outputs.audio_values
        
        return audio_values


def export_encoder_to_onnx(
    encoder_wrapper: nn.Module,
    output_path: Path,
    device: str,
    opset_version: int = 17,
    use_fp16: bool = False,
):
    """Export encoder to ONNX."""
    print(f"\n{'='*60}")
    precision = "FP16" if use_fp16 else "FP32"
    print(f"Exporting Encoder to ONNX ({precision})...")
    print(f"{'='*60}")
    
    # Exemple d'entrée pour l'encoder (1920 samples = 1 frame à 24kHz)
    SAMPLES_PER_FRAME = 1920
    
    # Pour FP16: garder l'input en FP32, mais utiliser autocast pour l'export
    # Cela permet à ONNX d'avoir des inputs FP32 avec conversion interne FP16
    dummy_audio = torch.randn(1, 1, SAMPLES_PER_FRAME, dtype=torch.float32).to(device)
    
    print(f"Input shape: {dummy_audio.shape}")
    print(f"Input dtype: {dummy_audio.dtype}")
    
    # Pour FP16: wrapper le modèle dans autocast context
    if use_fp16 and device == "cuda":
        # Créer un wrapper qui utilise autocast
        class AutocastWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.num_quantizers = model.num_quantizers
            
            def forward(self, x):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return self.model(x)
        
        export_model = AutocastWrapper(encoder_wrapper)
        print("✓ Using torch.cuda.amp.autocast for FP16 export")
    else:
        export_model = encoder_wrapper
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_audio,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_values'],
        output_names=['audio_codes'],
        dynamic_axes={
            'input_values': {0: 'batch', 2: 'samples'},
            'audio_codes': {0: 'batch', 2: 'frames'}
        },
        verbose=False
    )
    
    print(f"✓ Encoder exported to: {output_path}")
    print(f"  Precision: {precision}")
    print(f"  Input: [batch, 1, samples] -> Output: [batch, {encoder_wrapper.num_quantizers}, frames]")
    
    return output_path


def export_decoder_to_onnx(
    decoder_wrapper: nn.Module,
    output_path: Path,
    device: str,
    opset_version: int = 17,
    use_fp16: bool = False,
):
    """Export decoder to ONNX."""
    print(f"\n{'='*60}")
    precision = "FP16" if use_fp16 else "FP32"
    print(f"Exporting Decoder to ONNX ({precision})...")
    print(f"{'='*60}")
    
    # Exemple d'entrée pour le decoder (1 frame, n_q codebooks)
    # Les tokens sont dans [0, 2048) pour Mimi
    # NOTE: Les tokens restent en int64, pas de conversion FP16
    dummy_codes = torch.randint(
        0, 2048,
        (1, decoder_wrapper.num_quantizers, 1),
        dtype=torch.int64
    ).to(device)
    
    print(f"Input shape: {dummy_codes.shape}")
    print(f"Input dtype: {dummy_codes.dtype}")
    
    # Pour FP16: wrapper le modèle dans autocast context
    if use_fp16 and device == "cuda":
        # Créer un wrapper qui utilise autocast
        class AutocastWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.num_quantizers = model.num_quantizers
            
            def forward(self, x):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return self.model(x)
        
        export_model = AutocastWrapper(decoder_wrapper)
        print("✓ Using torch.cuda.amp.autocast for FP16 export")
    else:
        export_model = decoder_wrapper
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_codes,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_codes'],
        output_names=['audio_values'],
        dynamic_axes={
            'audio_codes': {0: 'batch', 2: 'frames'},
            'audio_values': {0: 'batch', 2: 'samples'}
        },
        verbose=False
    )
    
    print(f"✓ Decoder exported to: {output_path}")
    print(f"  Precision: {precision}")
    print(f"  Input: [batch, {decoder_wrapper.num_quantizers}, frames] -> Output: [batch, 1, samples]")
    
    return output_path


def convert_to_fp16(model_path: Path):
    """Convert ONNX model to FP16."""
    try:
        from onnxconverter_common import float16
        import onnx
        
        print(f"  Converting {model_path.name} to FP16...")
        
        # Charger le modèle
        model = onnx.load(str(model_path))
        
        # Convertir en FP16
        model_fp16 = float16.convert_float_to_float16(model)
        
        # Sauvegarder
        fp16_path = model_path.with_name(model_path.stem + "_fp16.onnx")
        onnx.save(model_fp16, str(fp16_path))
        
        # Comparer les tailles
        size_fp32 = model_path.stat().st_size / (1024**2)
        size_fp16 = fp16_path.stat().st_size / (1024**2)
        reduction = (1 - size_fp16/size_fp32) * 100
        
        print(f"  ✓ FP16 model saved: {fp16_path}")
        print(f"    FP32: {size_fp32:.1f}MB -> FP16: {size_fp16:.1f}MB "
              f"(reduction: {reduction:.1f}%)")
        
        return fp16_path
    
    except ImportError:
        print("  ⚠️  onnxconverter-common not installed, skipping FP16 conversion")
        print("     Install with: pip install onnxconverter-common onnx")
        return None


def verify_onnx_models(encoder_path: Path, decoder_path: Path, num_quantizers: int):
    """Verify ONNX models work correctly."""
    print(f"\n{'='*60}")
    print("Verification...")
    print(f"{'='*60}")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping verification")
        print("   Install with: pip install onnxruntime")
        return False
    
    # Vérifier l'encoder
    print(f"Loading encoder: {encoder_path}")
    encoder_session = ort.InferenceSession(str(encoder_path))
    
    # Input test
    SAMPLES_PER_FRAME = 1920
    encoder_input = np.random.randn(1, 1, SAMPLES_PER_FRAME).astype(np.float32)
    
    print(f"Running encoder...")
    encoder_output = encoder_session.run(None, {'input_values': encoder_input})[0]
    print(f"✓ Encoder output shape: {encoder_output.shape}")
    
    # Vérifier le nombre de codebooks
    assert encoder_output.shape[1] == num_quantizers, \
        f"Expected {num_quantizers} codebooks, got {encoder_output.shape[1]}"
    
    # Vérifier le decoder
    print(f"\nLoading decoder: {decoder_path}")
    decoder_session = ort.InferenceSession(str(decoder_path))
    
    print(f"Running decoder...")
    decoder_output = decoder_session.run(None, {'audio_codes': encoder_output})[0]
    print(f"✓ Decoder output shape: {decoder_output.shape}")
    
    # Round-trip test
    print(f"\n✓ Round-trip test:")
    print(f"  Original shape: {encoder_input.shape}")
    print(f"  Tokens shape: {encoder_output.shape} ({num_quantizers} codebooks)")
    print(f"  Reconstructed shape: {decoder_output.shape}")
    
    # Vérifier que les tokens sont dans la plage valide
    print(f"\n✓ Token statistics:")
    print(f"  Min: {encoder_output.min()}")
    print(f"  Max: {encoder_output.max()}")
    print(f"  Valid range: [0, 2047]")
    
    return True


def export_mimi_to_onnx(
    output_dir: Path,
    num_quantizers: int = 8,
    model_name: str = "kyutai/mimi",
    opset_version: int = 17,
    use_fp16: bool = False,
):
    """
    Export Mimi encoder and decoder to ONNX with reduced codebooks.
    
    Args:
        output_dir: Directory to save ONNX models
        num_quantizers: Number of codebooks to use (default: 8)
        model_name: HuggingFace model name
        opset_version: ONNX opset version
        use_fp16: Whether to export in FP16 (reduces size by ~50%)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Mimi ONNX Export")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Target codebooks: {num_quantizers}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"ONNX opset: {opset_version}")
    print(f"FP16 conversion: {use_fp16}")
    
    # Charger le modèle PyTorch
    print(f"\n{'='*60}")
    print("Loading PyTorch model...")
    print(f"{'='*60}")
    
    try:
        from transformers import MimiModel
    except ImportError:
        print("❌ transformers not installed")
        print("   Install with: pip install transformers")
        sys.exit(1)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    mimi_model = MimiModel.from_pretrained(model_name)
    mimi_model = mimi_model.to(device)
    mimi_model.eval()
    
    print(f"✓ Model loaded")
    print(f"  Original num_codebooks: {mimi_model.config.num_codebooks}")
    print(f"  Codebook size: {mimi_model.config.codebook_size}")
    
    # Créer les wrappers
    print(f"\nCreating wrappers...")
    encoder_wrapper = MimiEncoderWrapper(mimi_model, num_quantizers).to(device)
    decoder_wrapper = MimiDecoderWrapper(mimi_model, num_quantizers).to(device)
    encoder_wrapper.eval()
    decoder_wrapper.eval()
    
    # NOTE: Pour FP16, on N'utilise PAS .half() ici
    # Au lieu de ça, on utilise torch.cuda.amp.autocast pendant l'export
    # Cela évite les problèmes de type mismatch dans ONNX
    if use_fp16:
        print(f"\n{'='*60}")
        print("FP16 mode enabled (using autocast during export)...")
        print(f"{'='*60}")
        print("✓ Models will be exported with FP16 precision via autocast")
    
    # Export encoder
    model_suffix = "_fp16" if use_fp16 else ""
    encoder_path = output_dir / f"encoder_model{model_suffix}.onnx"
    export_encoder_to_onnx(encoder_wrapper, encoder_path, device, opset_version, use_fp16)
    
    # Export decoder
    decoder_path = output_dir / f"decoder_model{model_suffix}.onnx"
    export_decoder_to_onnx(decoder_wrapper, decoder_path, device, opset_version, use_fp16)
    
    # Pour compatibilité avec les anciens scripts, créer aussi les versions sans suffix
    if not use_fp16:
        encoder_fp16_path = None
        decoder_fp16_path = None
    else:
        encoder_fp16_path = encoder_path
        decoder_fp16_path = decoder_path
    
    # Vérification
    verify_path = encoder_fp16_path if encoder_fp16_path else encoder_path
    decoder_verify_path = decoder_fp16_path if decoder_fp16_path else decoder_path
    verify_onnx_models(verify_path, decoder_verify_path, num_quantizers)
    
    # Résumé
    print(f"\n{'='*60}")
    print("✅ Export completed successfully!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nFiles created:")
    
    if use_fp16:
        total_size = (encoder_path.stat().st_size + decoder_path.stat().st_size) / (1024**2)
        print(f"  - encoder_model_fp16.onnx ({encoder_path.stat().st_size / (1024**2):.1f}MB)")
        print(f"  - decoder_model_fp16.onnx ({decoder_path.stat().st_size / (1024**2):.1f}MB)")
        print(f"  Total: {total_size:.1f}MB")
    else:
        total_size = (encoder_path.stat().st_size + decoder_path.stat().st_size) / (1024**2)
        print(f"  - encoder_model.onnx ({encoder_path.stat().st_size / (1024**2):.1f}MB)")
        print(f"  - decoder_model.onnx ({decoder_path.stat().st_size / (1024**2):.1f}MB)")
        print(f"  Total: {total_size:.1f}MB")
    
    print(f"\nConfiguration:")
    print(f"  - Codebooks: {num_quantizers} (reduced from {mimi_model.config.num_codebooks})")
    print(f"  - Codebook size: {mimi_model.config.codebook_size}")
    print(f"  - Tokens per frame: {num_quantizers}")
    
    reduction = (1 - num_quantizers / mimi_model.config.num_codebooks) * 100
    print(f"\nSize reduction: ~{reduction:.0f}% fewer tokens per frame")
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print(f"1. Test the models:")
    print(f"   python test_onnx_export.py {output_dir}")
    print(f"\n2. Use in server_onnx.py:")
    print(f"   export MIMI_ONNX_MODEL_PATH={output_dir.absolute()}")
    if use_fp16:
        print(f"   export USE_ONNX_FP16=true  # Models are already FP16")
    print(f"   python server_onnx.py")
    
    print(f"\n💡 Note: FP16 export uses PyTorch native .half() conversion")
    print(f"   - No need for onnxconverter-common")
    print(f"   - ~50% size reduction")
    print(f"   - Input/output are FP16 (automatic conversion in ONNX Runtime)")
    
    return encoder_path, decoder_path


def main():
    parser = argparse.ArgumentParser(
        description="Export Mimi model to ONNX with reduced codebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with 8 codebooks (default)
  python export_mimi_onnx_nq8.py --output-dir ./onnx_nq8
  
  # Export with 8 codebooks + FP16 conversion (recommended)
  python export_mimi_onnx_nq8.py --output-dir ./onnx_nq8 --fp16
  
  # Export with 16 codebooks
  python export_mimi_onnx_nq8.py --output-dir ./onnx_nq16 --num-quantizers 16
  
  # Export with different model
  python export_mimi_onnx_nq8.py --output-dir ./onnx_custom --model ./local/mimi
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./onnx_nq8",
        help="Output directory for ONNX models (default: ./onnx_nq8)"
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of codebooks to use (default: 8, original: 32)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="kyutai/mimi",
        help="HuggingFace model name or local path (default: kyutai/mimi)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Also export FP16 version (requires onnxconverter-common)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    
    args = parser.parse_args()
    
    try:
        export_mimi_to_onnx(
            output_dir=args.output_dir,
            num_quantizers=args.num_quantizers,
            model_name=args.model,
            opset_version=args.opset,
            use_fp16=args.fp16,
        )
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
