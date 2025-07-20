#!/usr/bin/env python3
"""
Example: Using CHiME-6 Dataset with Whisper

This script demonstrates how to use the prepared CHiME-6 dataset
with OpenAI Whisper for speech recognition evaluation.

Usage:
    python example_chime6_usage.py --chime6_path /path/to/CHiME6 --model base
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    import whisper
except ImportError:
    print("Error: OpenAI Whisper is not installed.")
    print("Install it with: pip install openai-whisper")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CHiME6WhisperEvaluator:
    """Evaluates Whisper performance on CHiME-6 dataset."""
    
    def __init__(self, chime6_path: str, model_name: str = "base"):
        self.chime6_path = Path(chime6_path)
        self.model_name = model_name
        self.model = None
        
        # Validate CHiME-6 dataset
        self._validate_dataset()
        
    def _validate_dataset(self) -> None:
        """Validate CHiME-6 dataset structure."""
        if not self.chime6_path.exists():
            raise FileNotFoundError(f"CHiME-6 path does not exist: {self.chime6_path}")
            
        required_dirs = ['audio', 'transcriptions']
        for dir_name in required_dirs:
            dir_path = self.chime6_path / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
                
        # Check for manifest file
        manifest_path = self.chime6_path / 'binaural_manifest.json'
        if not manifest_path.exists():
            logger.warning(f"Binaural manifest not found: {manifest_path}")
            logger.warning("Will search for audio files directly")
            
    def load_model(self) -> None:
        """Load Whisper model."""
        logger.info(f"Loading Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name)
        logger.info("Model loaded successfully")
        
    def get_audio_files(self, split: str = "dev", limit: Optional[int] = None) -> List[Path]:
        """Get list of binaural audio files for a given split."""
        # Try to use manifest first
        manifest_path = self.chime6_path / 'binaural_manifest.json'
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                
            if split in manifest:
                audio_files = []
                for rel_path in manifest[split]:
                    full_path = self.chime6_path / 'audio' / split / rel_path
                    if full_path.exists():
                        audio_files.append(full_path)
                        
                if limit:
                    audio_files = audio_files[:limit]
                    
                return audio_files
        
        # Fallback: search for binaural files directly
        logger.info(f"Searching for binaural files in {split} split...")
        audio_dir = self.chime6_path / 'audio' / split
        
        if not audio_dir.exists():
            logger.warning(f"Audio directory not found: {audio_dir}")
            return []
            
        import glob
        pattern = str(audio_dir / "**" / "*_P??.wav")
        audio_files = [Path(f) for f in glob.glob(pattern, recursive=True)]
        
        if limit:
            audio_files = audio_files[:limit]
            
        return audio_files
    
    def transcribe_file(self, audio_path: Path, **kwargs) -> Dict:
        """Transcribe a single audio file."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        logger.debug(f"Transcribing: {audio_path.name}")
        
        try:
            result = self.model.transcribe(str(audio_path), **kwargs)
            return {
                'file': str(audio_path),
                'text': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'success': True,
                'error': None
            }
        except Exception as e:
            logger.error(f"Error transcribing {audio_path.name}: {e}")
            return {
                'file': str(audio_path),
                'text': '',
                'language': 'unknown',
                'success': False,
                'error': str(e)
            }
    
    def evaluate_split(self, split: str = "dev", limit: Optional[int] = None, 
                      save_results: bool = True, **transcribe_kwargs) -> List[Dict]:
        """Evaluate Whisper on a dataset split."""
        logger.info(f"Evaluating on {split} split...")
        
        # Get audio files
        audio_files = self.get_audio_files(split, limit)
        
        if not audio_files:
            logger.warning(f"No audio files found for {split} split")
            return []
            
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Transcribe files
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            logger.info(f"Processing {i}/{len(audio_files)}: {audio_path.name}")
            
            result = self.transcribe_file(audio_path, **transcribe_kwargs)
            results.append(result)
            
            # Print result
            if result['success']:
                print(f"\n📁 File: {audio_path.name}")
                print(f"🗣️  Text: {result['text']}")
                print(f"🌍 Language: {result['language']}")
            else:
                print(f"\n❌ Failed: {audio_path.name} - {result['error']}")
        
        # Save results
        if save_results:
            output_file = self.chime6_path / f"whisper_results_{split}_{self.model_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f"\nSummary: {successful}/{len(results)} files processed successfully")
        
        return results
    
    def compare_with_reference(self, results: List[Dict], split: str = "dev") -> None:
        """Compare Whisper results with reference transcriptions (if available)."""
        logger.info("Comparing with reference transcriptions...")
        
        # This is a placeholder for reference comparison
        # In a full implementation, you would:
        # 1. Load reference transcriptions from CHiME-6 JSON files
        # 2. Align Whisper outputs with references
        # 3. Compute WER (Word Error Rate) and other metrics
        
        logger.warning("Reference comparison not implemented in this example")
        logger.info("To implement reference comparison:")
        logger.info("1. Parse CHiME-6 transcription JSON files")
        logger.info("2. Align timestamps with audio segments")
        logger.info("3. Compute WER using libraries like jiwer")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper on CHiME-6 dataset"
    )
    parser.add_argument(
        "--chime6_path",
        required=True,
        help="Path to prepared CHiME-6 dataset directory"
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)"
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "eval"],
        help="Dataset split to evaluate (default: dev)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process (for testing)"
    )
    parser.add_argument(
        "--language",
        help="Force specific language (e.g., 'en' for English)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create evaluator
    evaluator = CHiME6WhisperEvaluator(args.chime6_path, args.model)
    
    # Load model
    evaluator.load_model()
    
    # Prepare transcription options
    transcribe_kwargs = {}
    if args.language:
        transcribe_kwargs['language'] = args.language
    
    # Run evaluation
    try:
        results = evaluator.evaluate_split(
            split=args.split,
            limit=args.limit,
            **transcribe_kwargs
        )
        
        # Optional: Compare with references
        # evaluator.compare_with_reference(results, args.split)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📊 Processed {len(results)} files")
        print(f"💾 Results saved to CHiME-6 directory")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()