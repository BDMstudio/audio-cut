# SeamlessSplitter 重构方案                                                                                                
                                                                                                                             
  ## 1. 问题诊断                                                                                                             
                                                                                                                             
  ### 1.1 现状                                                                                                               
  - `seamless_splitter.py`: 2678 行，严重违反 SRP                                                                            
  - `_process_hybrid_mdd_split()`: 450+ 行 (Plan A 未提取)                                                                   
  - `_process_pure_vocal_split()`: 400+ 行                                                                                   
  - **758 行重复代码**分布在 5 个处理方法中                                                                                  
                                                                                                                             
  ### 1.2 重复代码定位                                                                                                       
  | 模式 | 位置 | 重复次数 | 行数 |                                                                                          
  |-----|------|---------|------|                                                                                            
  | 节拍检测 | L767, L1138, L1536, L1783 | 4 | ~20 |                                                                         
  | 小节生成 | L1151, L1547, L1794 | 3 | ~36 |                                                                               
  | 小节能量 | L781, L1166, L1562, L1808 | 4 | ~80 |                                                                         
  | 导出逻辑 | 6 处 | 6 | ~240 |                                                                                             
  | 结果字典 | 5 处 | 5 | ~140 |                                                                                             
                                                                                                                             
  ### 1.3 隐藏状态依赖问题                                                                                                   
  ```python                                                                                                                  
  # 以下状态在方法间隐式传递，增加了复杂性:                                                                                  
  self._last_separation_result      # 用于但从未显式设置                                                                     
  self._last_guard_adjustments_raw  # 影响 layout refinement                                                                 
  self._last_suppressed_cut_points  # 影响 rescue splits                                                                     
  ```                                                                                                                        
                                                                                                                             
  ### 1.4 现有基础设施（可复用）                                                                                             
  - `TrackFeatureCache`: 已有 beat_times, rms_series, mdd_series, bpm_features                                               
  - `segment_layout_refiner`: 已有 Segment, micro-merge, beat-snap                                                           
  - `strategies/base.py`: 已有 SegmentationContext, SegmentationResult                                                       
  - `refine.py`: 已有 CutPoint, NMS, quiet guard                                                                             
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 2. 重构目标                                                                                                             
                                                                                                                             
  | 指标 | 当前 | 目标 |                                                                                                     
  |-----|------|------|                                                                                                      
  | seamless_splitter.py 行数 | 2678 | ~1200 (-55%) |                                                                        
  | 最大方法行数 | 450 | ~100 |                                                                                              
  | 代码重复 | 758 行 | ~0 |                                                                                                 
  | 策略类 | 2 (B/C) | 3 (A/B/C) |                                                                                           
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 3. 重构方案                                                                                                             
                                                                                                                             
  ### 3.1 新增模块结构                                                                                                       
                                                                                                                             
  ```                                                                                                                        
  src/audio_cut/analysis/                                                                                                    
  ├── features_cache.py             # 已有 TrackFeatureCache                                                                 
  ├── beat_analyzer.py              # [NEW] 节拍/能量分析 (基础设施层)                                                       
  └── __init__.py                   # 更新导出                                                                               
                                                                                                                             
  src/vocal_smart_splitter/core/                                                                                             
  ├── seamless_splitter.py          # 主编排器 (~1200行)                                                                     
  ├── strategies/                                                                                                            
  │   ├── base.py                   # + 共享工具函数                                                                         
  │   ├── mdd_start_strategy.py     # [NEW] Plan A                                                                           
  │   ├── beat_only_strategy.py     # Plan B (已有)                                                                          
  │   └── snap_to_beat_strategy.py  # Plan C (已有)                                                                          
  └── utils/                                                                                                                 
  ├── segment_exporter.py       # [NEW] 片段导出                                                                             
  └── result_builder.py         # [NEW] 结果字典构建                                                                         
  ```                                                                                                                        
                                                                                                                             
  **BeatAnalyzer 放置于 `audio_cut/analysis/` 的理由:**                                                                      
  1. 与 `TrackFeatureCache` 集成，复用已有的 beat_times 和 bpm_features                                                      
  2. 跨模式复用 (librosa_onset, hybrid_mdd 都需要)                                                                           
  3. 遵循 CPU/GPU 后备模式                                                                                                   
                                                                                                                             
  ### 3.2 关键提取                                                                                                           
                                                                                                                             
  #### A. MddStartStrategy (Plan A) [HIGH PRIORITY]                                                                          
  **来源**: `_process_hybrid_mdd_split()` lines 1105-1282 (核心逻辑)                                                         
  **职责**: MDD 起点 + 节拍终点策略                                                                                          
                                                                                                                             
  **提取边界分析:**                                                                                                          
  - Lines 1105-1282: 核心策略逻辑 (节拍检测 + 能量分析 + 切点合并) → 提取到 Strategy                                         
  - Lines 1283-1376: 分类 + 布局优化 (共享逻辑) → 保留在编排器                                                               
  - Lines 1377-1479: 导出 + 结果构建 (共享逻辑) → 提取到 Exporter/Builder                                                    
                                                                                                                             
  ```python                                                                                                                  
  class MddStartStrategy(SegmentationStrategy):                                                                              
  @property                                                                                                                  
  def name(self) -> str:                                                                                                     
  return "mdd_start"                                                                                                         
                                                                                                                             
  def generate_cut_points(self, context: SegmentationContext) -> SegmentationResult:                                         
  # 1. 从 context 获取高能量小节 (复用 base.py 工具函数)                                                                     
  high_energy_bars = identify_high_energy_bars(                                                                              
  context.bar_energies, context.energy_threshold                                                                             
  )                                                                                                                          
                                                                                                                             
  # 2. 在高能量区生成节拍切点                                                                                                
  beat_cut_times = self._generate_beat_cuts(...)                                                                             
                                                                                                                             
  # 3. 合并 MDD + 节拍切点 (带预过滤)                                                                                        
  final_cuts, cut_sources = deduplicate_and_convert_cuts(                                                                    
  context.mdd_cut_points_samples,                                                                                            
  beat_cut_samples,                                                                                                          
  snap_tolerance_samples,                                                                                                    
  )                                                                                                                          
                                                                                                                             
  # 4. 构建 lib_flags                                                                                                        
  lib_flags = build_lib_flags(final_cuts, cut_sources)                                                                       
                                                                                                                             
  return SegmentationResult(final_cuts, lib_flags, metadata={...})                                                           
  ```                                                                                                                        
                                                                                                                             
  **消除代码**: ~180 行 (核心逻辑)                                                                                           
                                                                                                                             
  #### B. BeatAnalyzer [HIGH PRIORITY]                                                                                       
  **文件**: `src/audio_cut/analysis/beat_analyzer.py`                                                                        
  **来源**: 4 处重复的节拍检测代码 (L767, L1138, L1536, L1783)                                                               
  **职责**: 封装 librosa 节拍/能量分析，集成 TrackFeatureCache                                                               
                                                                                                                             
  ```python                                                                                                                  
  # src/audio_cut/analysis/beat_analyzer.py                                                                                  
                                                                                                                             
  @dataclass                                                                                                                 
  class BeatAnalysisResult:                                                                                                  
  tempo: float                   # BPM                                                                                       
  beat_times: np.ndarray         # 拍点时间                                                                                  
  bar_times: np.ndarray          # 小节边界                                                                                  
  bar_duration: float            # 小节时长                                                                                  
  bar_energies: List[float]      # 每小节能量                                                                                
  high_energy_bars: Set[int]     # 高能量小节索引                                                                            
  energy_threshold: float        # 能量阈值                                                                                  
                                                                                                                             
  def analyze_beats(                                                                                                         
  audio: np.ndarray,                                                                                                         
  sr: int,                                                                                                                   
  *,                                                                                                                         
  hop_length: int = 512,                                                                                                     
  time_signature: int = 4,                                                                                                   
  energy_percentile: float = 70.0,                                                                                           
  feature_cache: Optional[TrackFeatureCache] = None,                                                                         
  ) -> BeatAnalysisResult:                                                                                                   
  """                                                                                                                        
  统一的节拍/能量分析入口。                                                                                                  
                                                                                                                             
  优先级:                                                                                                                    
  1. 从 feature_cache.beat_times 获取已计算的节拍                                                                            
  2. 从 feature_cache.bpm_features.main_bpm 获取 BPM                                                                         
  3. 回退到 librosa.beat.beat_track() 重新计算                                                                               
  """                                                                                                                        
  # 1. 获取节拍数据 (优先复用缓存)                                                                                           
  if feature_cache is not None and feature_cache.beat_times is not None:                                                     
  beat_times = feature_cache.beat_times                                                                                      
  tempo = feature_cache.bpm_features.main_bpm                                                                                
  else:                                                                                                                      
  tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)                                        
  tempo = float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0])                                                 
  beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)                                             
                                                                                                                             
  # 2. 生成小节边界                                                                                                          
  bar_times = _generate_bar_boundaries(beat_times, len(audio)/sr, time_signature)                                            
                                                                                                                             
  # 3. 计算小节能量                                                                                                          
  bar_energies = _compute_bar_energies(audio, sr, bar_times, hop_length)                                                     
                                                                                                                             
  # 4. 识别高能量小节                                                                                                        
  energy_threshold = float(np.percentile(bar_energies, energy_percentile)) if bar_energies else 0.0                          
  high_energy_bars = {i for i, e in enumerate(bar_energies) if e >= energy_threshold}                                        
                                                                                                                             
  return BeatAnalysisResult(...)                                                                                             
  ```                                                                                                                        
                                                                                                                             
  **消除代码**: ~100 行 (4 处 × 25 行)                                                                                       
                                                                                                                             
  #### C. SegmentExporter [MEDIUM PRIORITY]                                                                                  
  **来源**: 6 处重复的导出逻辑                                                                                               
  **职责**: 统一片段导出                                                                                                     
                                                                                                                             
  ```python                                                                                                                  
  @dataclass                                                                                                                 
  class ExportResult:                                                                                                        
  saved_files: List[str]                                                                                                     
  mix_segment_files: List[str]                                                                                               
  vocal_segment_files: List[str]                                                                                             
  full_vocal_file: Optional[str]                                                                                             
  full_instrumental_file: Optional[str]                                                                                      
                                                                                                                             
  class SegmentExporter:                                                                                                     
  def export_all(                                                                                                            
  self,                                                                                                                      
  segments: List[np.ndarray],                                                                                                
  vocal_segments: Optional[List[np.ndarray]],                                                                                
  vocal_track: Optional[np.ndarray],                                                                                         
  instrumental_track: Optional[np.ndarray],                                                                                  
  output_dir: str,                                                                                                           
  input_name: str,                                                                                                           
  mode: str,                                                                                                                 
  segment_vocal_flags: List[bool],                                                                                           
  lib_flags: Optional[List[bool]] = None,                                                                                    
  lib_suffix: str = "_lib",                                                                                                  
  export_flags: Set[str],                                                                                                    
  duration_map: Dict[int, float],                                                                                            
  sample_rate: int,                                                                                                          
  export_format: str,                                                                                                        
  export_options: Dict,                                                                                                      
  ) -> ExportResult:                                                                                                         
  # 统一处理 mix_segments, vocal_segments, full_vocal, full_instrumental                                                     
  ```                                                                                                                        
                                                                                                                             
  **消除代码**: ~180 行 (6 处 × 30 行)                                                                                       
                                                                                                                             
  #### D. ResultBuilder [MEDIUM PRIORITY]                                                                                    
  **来源**: 5 处重复的结果字典构建                                                                                           
  **职责**: 构建标准化结果                                                                                                   
                                                                                                                             
  ```python                                                                                                                  
  class ResultBuilder:                                                                                                       
  def build_base(                                                                                                            
  self,                                                                                                                      
  method: str,                                                                                                               
  segments: List[np.ndarray],                                                                                                
  cut_points_samples: List[int],                                                                                             
  segment_vocal_flags: List[bool],                                                                                           
  export_result: ExportResult,                                                                                               
  processing_time: float,                                                                                                    
  input_path: str,                                                                                                           
  output_dir: str,                                                                                                           
  sample_rate: int,                                                                                                          
  ) -> Dict[str, Any]:                                                                                                       
  # 构建所有方法共享的 15+ 个基础字段                                                                                        
                                                                                                                             
  def add_hybrid_metadata(                                                                                                   
  self,                                                                                                                      
  result: Dict,                                                                                                              
  lib_flags: List[bool],                                                                                                     
  hybrid_config: Dict,                                                                                                       
  beat_analysis: BeatAnalysisResult,                                                                                         
  ) -> Dict[str, Any]:                                                                                                       
  # 添加 hybrid_mdd 特有字段                                                                                                 
                                                                                                                             
  def add_separation_metadata(                                                                                               
  self,                                                                                                                      
  result: Dict,                                                                                                              
  separation_result: Any,                                                                                                    
  ) -> Dict[str, Any]:                                                                                                       
  # 添加 backend_used, separation_confidence, gpu_meta                                                                       
  ```                                                                                                                        
                                                                                                                             
  **消除代码**: ~100 行 (5 处 × 20 行)                                                                                       
                                                                                                                             
  #### E. 共享工具函数 (base.py) [LOW PRIORITY]                                                                              
  **来源**: B/C 策略中重复的工具函数                                                                                         
                                                                                                                             
  ```python                                                                                                                  
  # 添加到 base.py                                                                                                           
                                                                                                                             
  def identify_high_energy_bars(                                                                                             
  bar_energies: List[float],                                                                                                 
  energy_threshold: float,                                                                                                   
  ) -> Set[int]:                                                                                                             
  """识别高能量小节 (在 B/C 策略中重复)"""                                                                                   
                                                                                                                             
  def deduplicate_and_convert_cuts(                                                                                          
  cut_with_flags: List[Tuple[float, bool]],                                                                                  
  sample_rate: int,                                                                                                          
  audio_len: int,                                                                                                            
  ) -> Tuple[List[int], List[bool]]:                                                                                         
  """去重并转换切点到样本索引 (在 B/C 策略中重复 ~50 行)"""                                                                  
  ```                                                                                                                        
                                                                                                                             
  **消除代码**: ~100 行                                                                                                      
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 4. 重构后的 seamless_splitter.py 结构                                                                                   
                                                                                                                             
  ```python                                                                                                                  
  class SeamlessSplitter:                                                                                                    
  def __init__(self, sample_rate: int = 44100):                                                                              
  self.beat_analyzer = BeatAnalyzer()                                                                                        
  self.exporter = SegmentExporter()                                                                                          
  self.result_builder = ResultBuilder()                                                                                      
  self.strategies = {                                                                                                        
  'mdd_start': MddStartStrategy(),                                                                                           
  'beat_only': BeatOnlyStrategy(),                                                                                           
  'snap_to_beat': SnapToBeatStrategy(),                                                                                      
  }                                                                                                                          
                                                                                                                             
  def _process_hybrid_mdd_split(self, ...):                                                                                  
  # 1. 加载音频 & 分离人声 (~20 行)                                                                                          
  # 2. 节拍分析 (~5 行)                                                                                                      
  beat_result = self.beat_analyzer.analyze(...)                                                                              
                                                                                                                             
  # 3. 构建上下文 & 调用策略 (~10 行)                                                                                        
  context = SegmentationContext(...)                                                                                         
  strategy = self.strategies[lib_alignment]                                                                                  
  seg_result = strategy.generate_cut_points(context)                                                                         
                                                                                                                             
  # 4. 分类 & 导出 (~15 行)                                                                                                  
  segment_vocal_flags = self._classify_segments_vocal_presence(...)                                                          
  export_result = self.exporter.export_all(...)                                                                              
                                                                                                                             
  # 5. 构建结果 (~5 行)                                                                                                      
  return self.result_builder.build_base(...).add_hybrid_metadata(...)                                                        
                                                                                                                             
  # 总计: ~55 行 (原 ~450 行)                                                                                                
  ```                                                                                                                        
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 5. 实施步骤                                                                                                             
                                                                                                                             
  ### Phase 1: 提取 MddStartStrategy (P0)                                                                                    
  1. 创建 `strategies/mdd_start_strategy.py`                                                                                 
  2. 从 `_process_hybrid_mdd_split()` 提取 lines 1105-1289 的核心逻辑                                                        
  3. 更新 `_process_hybrid_mdd_split()` 调用新策略                                                                           
  4. 验证: 运行 `pytest tests/integration/test_pipeline_v2_valley.py`                                                        
                                                                                                                             
  ### Phase 2: 提取 BeatAnalyzer (P1)                                                                                        
  1. 创建 `analyzers/beat_analyzer.py`                                                                                       
  2. 整合 TrackFeatureCache 复用逻辑                                                                                         
  3. 替换 4 处重复的节拍检测代码                                                                                             
  4. 验证: 运行 hybrid_mdd 模式测试                                                                                          
                                                                                                                             
  ### Phase 3: 提取 SegmentExporter (P2)                                                                                     
  1. 创建 `exporters/segment_exporter.py`                                                                                    
  2. 合并 `_save_segments()` 和 `_save_segments_with_lib_suffix()`                                                           
  3. 替换 6 处导出逻辑                                                                                                       
  4. 验证: 检查输出文件命名正确                                                                                              
                                                                                                                             
  ### Phase 4: 提取 ResultBuilder (P3)                                                                                       
  1. 创建 `exporters/result_builder.py`                                                                                      
  2. 替换 5 处结果字典构建                                                                                                   
  3. 验证: 检查 API 返回结构不变                                                                                             
                                                                                                                             
  ### Phase 5: 共享工具函数 (P4)                                                                                             
  1. 更新 `strategies/base.py` 添加共享函数                                                                                  
  2. 重构 B/C 策略使用共享函数                                                                                               
  3. 验证: 运行所有策略测试                                                                                                  
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 6. 验证清单                                                                                                             
                                                                                                                             
  ### 功能验证                                                                                                               
  ```bash                                                                                                                    
  # 快速回归                                                                                                                 
  pytest -m "not slow and not gpu" --cov=src --cov-report=term-missing                                                       
                                                                                                                             
  # hybrid_mdd 完整测试                                                                                                      
  python run_splitter.py input/test.mp3 --mode hybrid_mdd                                                                    
                                                                                                                             
  # 三种策略对比                                                                                                             
  python run_splitter.py input/test.mp3 --mode hybrid_mdd  # Plan A (default)                                                
  # 修改 unified.yaml: lib_alignment: beat_only                                                                              
  python run_splitter.py input/test.mp3 --mode hybrid_mdd  # Plan B                                                          
  # 修改 unified.yaml: lib_alignment: snap_to_beat                                                                           
  python run_splitter.py input/test.mp3 --mode hybrid_mdd  # Plan C                                                          
  ```                                                                                                                        
                                                                                                                             
  ### 新增单元测试                                                                                                           
                                                                                                                             
  #### BeatAnalyzer 测试 (`tests/unit/test_beat_analyzer.py`)                                                                
  | 测试用例 | 描述 |                                                                                                        
  |---------|------|                                                                                                         
  | `test_basic_detection` | 验证合成 120 BPM 信号的 tempo, beat_times |                                                     
  | `test_bar_generation_4_4_time` | 验证 4/4 拍小节边界 |                                                                   
  | `test_bar_generation_3_4_time` | 验证 3/4 拍处理 |                                                                       
  | `test_energy_threshold` | 验证百分位阈值计算 |                                                                           
  | `test_high_energy_bar_detection` | 确认高能量小节正确识别 |                                                              
  | `test_feature_cache_reuse` | 确保优先复用 cache.beat_times |                                                             
  | `test_fallback_short_audio` | 处理 < 4 拍的短音频 |                                                                      
                                                                                                                             
  #### MddStartStrategy 测试 (`tests/unit/test_mdd_start_strategy.py`)                                                       
  | 测试用例 | 描述 |                                                                                                        
  |---------|------|                                                                                                         
  | `test_basic_flow` | 验证 MDD + beat 合并生成切点 |                                                                       
  | `test_pre_filtering` | 确保 min_segment_s 约束被遵守 |                                                                   
  | `test_duplicate_detection` | 验证 snap_to_pause_ms 去重 |                                                                
  | `test_lib_flag_assignment` | 确认 lib flags 仅在 beat-aligned 末端 |                                                     
  | `test_empty_mdd_cuts` | MDD 无停顿时的处理 |                                                                             
  | `test_no_high_energy_bars` | 全低能量小节的行为 |                                                                        
                                                                                                                             
  #### SegmentExporter 测试 (`tests/unit/test_segment_exporter.py`)                                                          
  | 测试用例 | 描述 |                                                                                                        
  |---------|------|                                                                                                         
  | `test_basic_export` | 验证 WAV 输出命名正确 |                                                                            
  | `test_lib_suffix` | 确认 lib_flag=True 时添加 `_lib` |                                                                   
  | `test_duration_in_filename` | 验证时长后缀格式 |                                                                         
  | `test_subdir_creation` | 验证子目录创建 |                                                                                
                                                                                                                             
  ### 接口兼容性验证                                                                                                         
  ```python                                                                                                                  
  # API 返回结构必须不变 (这些字段被 api.py _build_manifest 使用)                                                            
  from audio_cut.api import separate_and_segment                                                                             
  result = separate_and_segment(input_uri="...", export_dir="...", mode="hybrid_mdd")                                        
                                                                                                                             
  # 必须存在的字段 (Never Break)                                                                                             
  assert 'success' in result                                                                                                 
  assert 'cut_points_sec' in result                                                                                          
  assert 'cut_points_samples' in result                                                                                      
  assert 'segment_labels' in result                                                                                          
  assert 'segment_durations' in result                                                                                       
  assert 'mix_segment_files' in result                                                                                       
  assert 'vocal_segment_files' in result                                                                                     
  assert 'segment_layout_applied' in result                                                                                  
  assert 'guard_shift_stats' in result                                                                                       
                                                                                                                             
  # hybrid_mdd 特有字段                                                                                                      
  assert 'segment_lib_flags' in result                                                                                       
  assert 'hybrid_config' in result                                                                                           
  ```                                                                                                                        
                                                                                                                             
  ### 性能基线                                                                                                               
  - 处理时间变化 < 5%                                                                                                        
  - 内存使用变化 < 10%                                                                                                       
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 7. 风险与缓解                                                                                                           
                                                                                                                             
  | 风险 | 缓解措施 |                                                                                                        
  |-----|---------|                                                                                                          
  | 策略提取破坏 lib_flags 逻辑 | 添加单元测试验证 lib_flags 数量 = num_segments |                                           
  | BeatAnalyzer 与 TrackFeatureCache 集成问题 | 优先从 cache 读取，回退到 librosa 计算 |                                    
  | 导出逻辑变更影响文件命名 | 添加文件名格式回归测试 |                                                                      
  | 结果字典字段遗漏 | 添加 JSON schema 验证测试 |                                                                           
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 8. 预期收益                                                                                                             
                                                                                                                             
  - **代码行数**: 2678 → ~1200 (-55%)                                                                                        
  - **最大方法**: 450 行 → ~100 行                                                                                           
  - **重复代码**: 758 行 → ~0                                                                                                
  - **可测试性**: 每个 Analyzer/Strategy 可独立单测                                                                          
  - **可维护性**: 修改节拍分析只需改 BeatAnalyzer 一处                                                                       
                                                                                                                             
  ---                                                                                                                        
                                                                                                                             
  ## 9. 关键文件清单                                                                                                         
                                                                                                                             
  ### 需要修改的文件                                                                                                         
  | 文件 | 修改类型 | 说明 |                                                                                                 
  |-----|---------|------|                                                                                                   
  | `src/vocal_smart_splitter/core/seamless_splitter.py` | 重构 | 主要重构目标，从 2678 行降至 ~1200 行 |                    
  | `src/vocal_smart_splitter/core/strategies/base.py` | 扩展 | 添加共享工具函数 |                                           
  | `src/vocal_smart_splitter/core/strategies/beat_only_strategy.py` | 重构 | 使用 base.py 共享函数 |                        
  | `src/vocal_smart_splitter/core/strategies/snap_to_beat_strategy.py` | 重构 | 使用 base.py 共享函数 |                     
  | `src/audio_cut/analysis/__init__.py` | 扩展 | 导出 BeatAnalyzer |                                                        
                                                                                                                             
  ### 需要新建的文件                                                                                                         
  | 文件 | 说明 |                                                                                                            
  |-----|------|                                                                                                             
  | `src/audio_cut/analysis/beat_analyzer.py` | 节拍/能量分析 |                                                              
  | `src/vocal_smart_splitter/core/strategies/mdd_start_strategy.py` | Plan A 策略 |                                         
  | `src/vocal_smart_splitter/utils/segment_exporter.py` | 片段导出 |                                                        
  | `src/vocal_smart_splitter/utils/result_builder.py` | 结果字典构建 |                                                      
  | `tests/unit/test_beat_analyzer.py` | BeatAnalyzer 测试 |                                                                 
  | `tests/unit/test_mdd_start_strategy.py` | MddStartStrategy 测试 |                                                        
  | `tests/unit/test_segment_exporter.py` | SegmentExporter 测试 |