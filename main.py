"""
Audio Style Projection — CLI Entry Point

Usage:
    python main.py --base samples/source_clouds.wav \
                   --projected samples/target_elvis.wav \
                   --output outputs/result.wav \
                   --alpha 0.7 \
                   --preset balanced
"""

import argparse
import sys
from pathlib import Path

# allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import transform_audio, PRESETS


def main():
    parser = argparse.ArgumentParser(
        description="🎵 Audio Style Projection — make one sound 'perform' another",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py --base samples/source_clouds.wav --projected samples/target_elvis.wav
  python main.py --base rain.wav --projected song.wav --alpha 0.5 --preset more-source
  python main.py --base fan.wav --projected humming.wav --preset more-singing
""",
    )
    parser.add_argument(
        "--base", required=True,
        help="Path to the base / source audio (clouds, wind, rain …)",
    )
    parser.add_argument(
        "--projected", required=True,
        help="Path to the projected / target audio (singing, melody …)",
    )
    parser.add_argument(
        "--output", default="outputs/output.wav",
        help="Where to save the result (default: outputs/output.wav)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7,
        help="Blend strength 0.0–1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--preset", default="balanced",
        choices=list(PRESETS.keys()),
        help="Quality preset (default: balanced)",
    )
    parser.add_argument(
        "--sr", type=int, default=22050,
        help="Sample rate for processing (default: 22050)",
    )
    parser.add_argument(
        "--engine", default="dsp", choices=["dsp", "neural"],
        help="Processing engine to use. 'dsp' for classic algorithm, 'neural' for DDSP (requires --model).",
    )
    parser.add_argument(
        "--model", help="Path to the trained neural voice model (.pt file) if engine is 'neural'",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  🎵 Audio Style Projection")
    print("=" * 60)
    print(f"  Base      : {args.base}")
    print(f"  Projected : {args.projected}")
    print(f"  Output    : {args.output}")
    print(f"  Alpha     : {args.alpha}")
    print(f"  Preset    : {args.preset}")
    print(f"  SR        : {args.sr}")
    print(f"  Engine    : {args.engine}")
    if args.engine == "neural":
        print(f"  Model     : {args.model}")
    print("=" * 60)

    if args.engine == "neural":
        if not args.model:
            print("❌ Error: --model is required when --engine=neural")
            sys.exit(1)
        from neural_pipeline import neural_transform
        neural_transform(
            model_path=args.model,
            projected_path=args.projected,
            output_path=args.output,
            sr=args.sr
        )
    else:
        transform_audio(
            base_path=args.base,
            projected_path=args.projected,
            output_path=args.output,
            alpha=args.alpha,
            preset=args.preset,
            sr=args.sr,
        )

    print("\n✅ Transformation complete!  Listen to:", args.output)


if __name__ == "__main__":
    main()
