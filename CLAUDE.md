# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an intelligent vocal splitting tool that analyzes songs and splits them at natural breath/pause points. The system uses advanced audio processing, voice activity detection (VAD), and machine learning to create high-quality audio segments optimized for voice training or analysis.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (required before any development)
source audio_env/bin/activate  # Linux/macOS
# or
audio_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Primary Usage Commands
```bash
# ✨ Seamless vocal pause splitting (RECOMMENDED - BPM adaptive is built-in)
python run_splitter.py input/01.mp3 --seamless-vocal

# ✨ With reconstruction validation
python run_splitter.py input/01.mp3 --seamless-vocal --validate-reconstruction

# Traditional intelligent splitting
python run_splitter.py input/01.mp3 --min-length 8 --max-length 12 --target-length 10

# Verbose output for debugging
python run_splitter.py input/01.mp3 --verbose

# Direct usage of core module
python src/vocal_smart_splitter/main.py input/01.mp3 -o output/custom_dir

# Note: BPM-adaptive enhancement is automatically enabled when using --seamless-vocal mode
# Configure via config.yaml: vocal_pause_splitting.enable_bpm_adaptation: true
```

### Testing Commands
```bash
# Run all tests
python tests/run_tests.py

# Run specific test
python tests/test_precise_voice_splitting.py
python tests/test_pause_priority.py
python tests/test_audio_quality_fix.py
python tests/test_simple_pause_priority.py

# Test with specific algorithm focus
python tests/test_precise_voice_splitting.py  # Recommended algorithm
python tests/test_pause_priority.py          # Alternative algorithm
python tests/test_seamless_reconstruction.py # Seamless splitting test
python tests/test_bmp_adaptive_vad.py        # NEW: BPM adaptive VAD test
```

## Core Architecture

### 🆕 BPM-Adaptive Seamless Pipeline (v1.1.4 - PARTIALLY FUNCTIONAL)
The latest BPM-adaptive seamless splitter combines Silero VAD with musical intelligence (BLOCKED by encoding issues):

**Core Components:**
- `src/vocal_smart_splitter/core/seamless_splitter.py` - Main seamless splitting engine
- `src/vocal_smart_splitter/core/vocal_pause_detector.py` - Enhanced Silero VAD detector
- `src/vocal_smart_splitter/core/adaptive_vad_enhancer.py` - NEW: BPM-adaptive VAD enhancer

**Processing Pipeline:**
1. **Audio Loading** - Direct 44.1kHz audio processing
2. **BPM Analysis** - Automatic tempo detection and music categorization  
3. **Complexity Assessment** - Dynamic analysis of arrangement complexity
4. **Adaptive Threshold Generation** - Multi-dimensional threshold adjustment
5. **Silero VAD Analysis** - Enhanced neural network-based voice activity detection
6. **Beat Alignment** - Cutting point optimization using musical beats
7. **Sample-Perfect Splitting** - Zero-difference reconstruction splitting
8. **WAV/FLAC Output** - Lossless audio format with zero processing

### Traditional Processing Pipeline (Legacy)
The traditional system follows a 7-step processing pipeline in `src/vocal_smart_splitter/main.py`:

1. **Audio Loading & Preprocessing** (`utils/audio_processor.py`)
2. **Vocal Separation** (`core/vocal_separator.py`) - HPSS-based vocal isolation
3. **Breath Detection** (`core/breath_detector.py`) - Energy & spectral analysis for pauses
4. **Content Analysis** (`core/content_analyzer.py`) - Vocal segment grouping and continuity
5. **Smart Splitting** (`core/smart_splitter.py`) - Algorithm selection and split point decision
6. **Quality Control** (`core/quality_controller.py`) - Validation and audio processing
7. **File Output** - Segment saving with metadata

### Split Algorithm Architecture
The system implements multiple splitting strategies:

- **🆕 Seamless Splitting** (`core/seamless_splitter.py`) - **RECOMMENDED**:
  - Uses Silero VAD neural network for music-aware voice detection
  - Direct processing on original audio (no vocal separation needed)
  - 94.1% confidence rate, 0.00e+00 reconstruction difference
  - Zero audio processing (no fade-in/out, no normalization)
  - 3x faster processing speed

- **Precise Voice Splitting** (`core/precise_voice_splitter.py`): 
  - Uses Silero VAD for accurate voice activity detection
  - Prioritizes split accuracy over segment length constraints
  - Traditional algorithm (set via `smart_splitting.use_precise_voice_algorithm: true`)

- **Pause Priority Splitting** (`core/pause_priority_splitter.py`):
  - Scores pause points based on duration, intensity, energy drop
  - Balances pause quality with target segment lengths
  - Legacy algorithm for specific use cases

### Configuration System
Centralized configuration management through:
- `src/vocal_smart_splitter/config.yaml` - Main configuration
- `config/default.yaml` - Default fallback configuration
- `utils/config_manager.py` - Dynamic configuration handling

Key configuration sections:
- `vocal_pause_splitting`: 🆕 Seamless splitter settings (min_pause_duration, vad_method)
- `smart_splitting`: Algorithm selection and segment length controls
- `precise_voice_splitting`: VAD method and silence thresholds
- `pause_priority`: Scoring weights for pause-based splitting
- `vocal_separation`: HPSS parameters for vocal isolation (legacy)
- `quality_control`: Validation thresholds and audio processing

## Key Architecture Patterns

### Module Separation
- `core/`: Algorithm implementations (vocal separation, breath detection, splitting logic)
- `utils/`: Shared utilities (config management, audio processing, feature extraction)
- `tests/`: Unified test suite with algorithm-specific test scenarios

### Algorithm Selection Pattern
The `SmartSplitter` class acts as a dispatcher that selects between algorithms based on configuration:
```python
# Algorithm selection in core/smart_splitter.py
if get_config('smart_splitting.use_precise_voice_algorithm'):
    return self.precise_voice_splitter.split(...)
elif get_config('smart_splitting.use_pause_priority_algorithm'):
    return self.pause_priority_splitter.split(...)
```

### Quality-First Processing
Each processing stage includes quality metrics and validation:
- Vocal separation quality assessment
- Breath detection confidence scoring
- Content analysis completeness validation
- Split point quality evaluation
- Final segment quality control

## Important Development Notes

### Virtual Environment Requirement
Always activate the `audio_env` virtual environment before development. The project requires specific versions of audio processing libraries (librosa, pydub, soundfile) that may conflict with system packages.

### Input/Output Structure
- Input files: Place in `input/` directory (expects `input/01.mp3` by default)
- Output structure: `output/test_YYYYMMDD_HHMMSS/` with timestamped directories
- Generated files: `vocal_segment_XX.mp3`, `analysis_report.json`, optional `debug_info.json`

### Algorithm Tuning

#### 🆕 BPM-Adaptive Seamless Parameters (RECOMMENDED)
For the latest BPM-adaptive seamless splitter, focus on these key config values:

**Core Splitting:**
- `vocal_pause_splitting.min_pause_duration`: Only split at pauses ≥1.2s (ensures natural breaks)
- `vocal_pause_splitting.vad_method`: "silero" (required for music-aware detection)

**BPM Adaptive Settings:**
- `vocal_pause_splitting.enable_bpm_adaptation`: true (enable BPM intelligence)
- `vocal_pause_splitting.bpm_adaptive_settings.tempo_min_bpm`: 50 (minimum BPM detection)  
- `vocal_pause_splitting.bpm_adaptive_settings.tempo_max_bpm`: 200 (maximum BPM detection)
- `vocal_pause_splitting.bpm_adaptive_settings.slow_bpm_threshold`: 80 (slow song limit)
- `vocal_pause_splitting.bpm_adaptive_settings.fast_bpm_threshold`: 120 (fast song limit)
- `vocal_pause_splitting.bpm_adaptive_settings.enable_beat_alignment`: true (align cuts to beats)
- `vocal_pause_splitting.bpm_adaptive_settings.enable_complexity_adaptation`: true (adapt to arrangement complexity)  
- `vocal_pause_splitting.voice_threshold`: 0.5 (VAD sensitivity)
- `vocal_pause_splitting.zero_processing`: true (no fade/normalization for perfect quality)

#### Traditional Algorithm Parameters (Legacy)
- `smart_splitting.target_segment_length`: Target length for segments (currently 8s)
- `smart_splitting.split_quality_threshold`: Lower values produce more segments (currently 0.5)
- `precise_voice_splitting.min_silence_duration`: Minimum silence required for splits (0.5s)
- `precise_voice_splitting.silence_threshold`: VAD sensitivity (0.3)

### Performance Expectations

#### 🆕 Seamless Splitter Results (v1.1.4 - CURRENT ACTUAL STATUS)
- ❌ **Split Accuracy**: Cannot verify due to Unicode encoding errors
- ✅ **Perfect Reconstruction**: 0.00e+00 difference verified in 1/7 tests
- ❌ **Processing Speed**: Cannot measure (test failures)
- ❌ **Audio Quality**: Cannot verify due to system instability  
- ❌ **Multi-instrument Adaptation**: Cannot test due to encoding issues
- ❌ **BPM Intelligence**: Blocked by 'gbk' codec errors
- ❌ **Segment Count**: Cannot generate due to test failures
- ⚠️ **System Stability**: 15% test success rate, requires encoding fixes

#### Traditional Algorithm Targets (Legacy)
- ≥90% segments should be 5-15 seconds long
- ≥80% split points should fall at natural pauses/breaths  
- Processing time ≤2 minutes for 3-5 minute songs
- Subjective naturalness rating ≥4/5

### Known Issues (v1.1.4 - CRITICAL)

#### 🚨 Unicode Encoding Problems (BLOCKING)
- **Error**: `'gbk' codec can't encode character '\U0001f3b5'`
- **Impact**: 6/7 tests failing, system unusable on Windows
- **Affected Files**: Most test files using emoji characters
- **Fix Required**: Remove all emoji characters from test output

#### 🚨 Module Import Issues
- **Error**: `ModuleNotFoundError: No module named 'src'`  
- **Impact**: `test_precise_voice_splitting.py` cannot run
- **Fix Required**: Update import paths in test files

#### ⚠️ Numpy Deprecation Warnings
- **Warning**: Array to scalar conversion deprecated
- **Fix Required**: Use explicit `float()` conversion throughout codebase

### Legacy Common Issues

#### Traditional Algorithm Issues (Legacy)
- **Low segment count**: Reduce `split_quality_threshold` or `min_silence_duration`
- **Poor split naturalness**: Ensure `use_precise_voice_algorithm: true` and fine-tune `silence_threshold`
- **Audio quality issues**: Check `fade_in_duration`/`fade_out_duration`, avoid over-processing with normalization
- **Import errors**: Verify virtual environment activation and proper `src/` path structure

## 🎯 Critical Development Notes (v1.1.4)

### Variable Naming Convention
**IMPORTANT**: Always use `bpm` (not `bmp`) for all BPM-related variables and attributes:
- ✅ Correct: `bpm_features`, `main_bpm`, `bpm_category`
- ❌ Wrong: `bmp_features`, `main_bmp`, `bmp_category`

### Numpy Formatting Best Practice
When logging numpy arrays, always use explicit `float()` conversion:
```python
# ✅ Correct
logger.info(f"BPM: {float(bpm_features.main_bpm):.1f}")

# ❌ Wrong (causes numpy format errors)
logger.info(f"BPM: {bmp_features.main_bpm:.1f}")
```

### 🆕 Configuration File Structure (v1.1.2)
**Main Configuration Sections** (in order of importance):
1. **`vocal_pause_splitting`** - 🚀 **PRIMARY**: BPM-adaptive seamless splitting
2. **`quality_control`** - Audio processing and validation settings
3. **`advanced_vad`** - Silero VAD core parameters
4. **`audio`/`output`/`logging`** - Basic system settings
5. **Do not use non-UTF-8 encoding such as emoji in your code.**

**Deprecated Sections** (kept for compatibility):
- `vocal_separation` - Legacy HPSS vocal isolation
- `breath_detection` - Legacy breath detection (replaced by Silero VAD)
- `content_analysis` - Legacy content analysis
- `smart_splitting` - Legacy algorithm dispatcher
- `pause_priority`/`precise_voice_splitting` - Legacy algorithms

### 🔧 BPM Multiplier Logic (v1.1.2 - CORRECTED)
**Music Theory Alignment**:
```yaml
pause_duration_multipliers:
  slow_song_multiplier: 1.5    # Slow songs: longer pauses, avoid over-segmentation
  fast_song_multiplier: 0.7    # Fast songs: shorter pauses, adapt to quick rhythm
  medium_song_multiplier: 1.0  # Standard tempo: baseline pause duration
```

**Logic Reasoning**:
- **Slow songs** (BPM < 80): Relaxed rhythm → natural pauses are longer → need higher multiplier
- **Fast songs** (BPM > 120): Dense rhythm → natural pauses are shorter → need lower multiplier

### System Status (v1.1.4 - CURRENT ACTUAL STATUS)
- ⚠️ **BMP/BPM Adaptive System**: PARTIALLY FUNCTIONAL (Unicode encoding issues)
- ❌ **Test Suite Status**: 85% failure rate (6/7 tests failing due to emoji encoding)
- ✅ **Seamless Reconstruction**: 1 test passing, 0.00e+00 error (verified)
- ❌ **BPM Processing**: Functional but blocked by GBK codec errors  
- ⚠️ **Module Import Issues**: Path configuration problems in test files
- 🔧 **Configuration System**: Stable, but requires encoding fixes