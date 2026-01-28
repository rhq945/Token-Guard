import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 配置类 (GPU 默认配置)
# ==========================================
@dataclass
class TokenGuardConfig:
    # Token-level
    lambda_val: float = 0.6       # Eq. 3
    tau_token: float = 0.4        # Eq. 4
    
    # Segment-level
    alpha: float = 0.5            # Eq. 9
    beta: float = 0.3             # Eq. 9
    gamma: float = 0.2            # Eq. 9
    tau_seg_low: float = 0.55     
    tau_seg_high: float = 0.75    
    
    # Global
    tau_global: float = 0.7       
    delta_tau: float = 0.1        # Eq. 15
    
    # [关键] 默认使用 CUDA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SegmentArtifact:
    """
    存储用于全局计算的片段级中间产物。
    为了防止长推理链导致 GPU 显存 OOM，我们将不参与计算的向量暂存到 CPU RAM 中，
    计算时再移回 GPU。
    """
    text: str
    H_k: torch.Tensor             # 片段表示向量 (Compact vector)
    e_tilde_k: torch.Tensor       # 平均输入 Embedding
    f_norm: float                 # 置信度向量范数
    segment_score: float          # F_halu^seg
    token_scores: List[float]     # Token 分数
    evidence_score: float = 1.0   

@dataclass
class RefinementAdvice:
    needed: bool
    window: Optional[Tuple[int, int]] = None
    target_token_idx: int = -1

# ==========================================
# 2. TokenGuardScorer (GPU Optimized)
# ==========================================
class TokenGuardScorer:
    def __init__(self, model_path: str, config: TokenGuardConfig = TokenGuardConfig()):
        self.config = config
        self.device = config.device
        
        print(f"[TokenGuard-GPU] Loading model on {self.device}: {model_path}")
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. 加载 Model (GPU FP16)
        # device_map="auto" 会自动将模型铺设在 GPU 上
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, # GPU 加速标准精度
            device_map="auto"
        ).eval()
        
        # [关键修复] 调整词表大小以匹配 Tokenizer (防止 Qwen/Llama3 Index Error)
        if len(self.tokenizer) > self.model.config.vocab_size:
            print(f"⚠️ [GPU Init] Resizing embeddings: {self.model.config.vocab_size} -> {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        # 运行时状态
        self.current_threshold_adj = 0.0 

    def _get_model_outputs(self, input_ids):
        """
        在 GPU 上获取 Hidden States
        """
        # input_ids 必须已经在 GPU 上
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        
        # Hidden States: 倒数第二层 (L-1) -> 用于语义计算
        hidden_states = outputs.hidden_states[-2] 
        
        # Input Embeddings: 第一层 (L=0) -> 用于上下文调整
        input_embeddings = outputs.hidden_states[0]
        
        return outputs.logits, hidden_states, input_embeddings

    def initialize_anchor(self, context_text: str) -> torch.Tensor:
        """
        [Eq. 1] 计算初始锚点 (在 GPU 上计算)
        """
        # 1. Move to GPU
        inputs = self.tokenizer(
            context_text, 
            return_tensors="pt",
            truncation=True,
            max_length=4096 
        ).to(self.device) 
        
        # 2. Compute on GPU
        _, hiddens, _ = self._get_model_outputs(inputs.input_ids)
        
        # 3. Aggregate on GPU
        h_x = torch.mean(hiddens[0], dim=0) 
        return h_x

    def verify_candidates(self, context_text: str, candidate_texts: List[str], h_x: torch.Tensor) -> List[SegmentArtifact]:
            """
            [Eq. 2 - Eq. 9] 批量验证候选片段 (GPU 加速)
            """
            # [Safety] Empty check
            if not candidate_texts:
                return []

            results = []
            
            # 确保锚点在 GPU 上
            if h_x.device.type != self.device:
                h_x = h_x.to(self.device)

            # 预处理 Context
            ctx_inputs = self.tokenizer(
                context_text, 
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            ctx_len = ctx_inputs.input_ids.shape[1]

            for text in candidate_texts:
                if not text.strip():
                    continue
                    
                full_text = context_text + text
                
                # [GPU] Move input to device
                inputs = self.tokenizer(
                    full_text, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=8192
                ).to(self.device)
                
                input_ids = inputs.input_ids
                
                # 安全检查
                current_ctx_len = min(ctx_len, input_ids.shape[1] - 1)
                if current_ctx_len >= input_ids.shape[1]:
                    continue 

                # [GPU] Forward Pass
                logits, full_hiddens, full_embeds = self._get_model_outputs(input_ids)
                
                # [GPU] Slicing & Gathering
                seg_logits = logits[0, current_ctx_len-1 : -1, :] 
                seg_ids = input_ids[0, current_ctx_len:]
                
                log_probs = F.log_softmax(seg_logits, dim=-1)
                token_log_probs = log_probs.gather(1, seg_ids.unsqueeze(1)).squeeze(1)
                token_probs = torch.exp(token_log_probs)
                
                seg_hiddens = full_hiddens[0, current_ctx_len:, :]
                seg_embeds = full_embeds[0, current_ctx_len:, :]
                
                # [GPU] Token-Level Scoring Loop
                token_scores = []
                running_h_mean = h_x.clone() # Keep on GPU
                t_offset = 1
                
                # 这里的循环虽然是 Python 循环，但内部全是 GPU Tensor 操作
                for i in range(len(seg_hiddens)):
                    h_t = seg_hiddens[i]
                    p_t = token_probs[i].item() # 获取标量用于加权，不影响大局
                    
                    # GPU Cosine Similarity
                    sim = F.cosine_similarity(h_t, running_h_mean, dim=0).item()
                    
                    f_token = self.config.lambda_val * sim + (1 - self.config.lambda_val) * p_t
                    token_scores.append(f_token)
                    
                    # GPU Update
                    running_h_mean = (running_h_mean * t_offset + h_t) / (t_offset + 1)
                    t_offset += 1
                    
                token_scores_t = torch.tensor(token_scores, device=self.device) # GPU Tensor
                
                # [Debug] Print token score info for visibility
                if len(token_scores) > 0:
                    avg_tok = sum(token_scores) / len(token_scores)
                    # print(f"    [Debug] Text: {text[:30]}... | Avg Token Score: {avg_tok:.4f}")
                
                # [GPU] Segment Representation
                weights = F.softmax(token_scores_t, dim=0)
                H_k = torch.sum(weights.unsqueeze(1) * seg_hiddens, dim=0) # Weighted Sum on GPU
                e_tilde_k = torch.mean(seg_embeds, dim=0)
                f_norm = torch.norm(token_scores_t).item()
                
                # [GPU -> CPU Scalar] Scoring Components
                score_token = torch.sum(weights * token_scores_t).item()
                
                if len(seg_hiddens) > 1:
                    diffs = seg_hiddens[1:] - seg_hiddens[:-1]
                    smoothness = torch.norm(diffs, dim=1).mean().item()
                    score_const = max(0.0, 1.0 - smoothness)
                else:
                    score_const = 1.0
                    
                score_align = F.cosine_similarity(H_k, h_x, dim=0).item()
                
                f_seg = (self.config.alpha * score_token + 
                        self.config.beta * score_const + 
                        self.config.gamma * score_align)
                
                # [Memory Management]
                # 计算完成的向量 H_k 和 e_tilde_k 存入 Artifact。
                # 为了防止在长链推理中显存堆积，这里将它们移至 CPU (.cpu())。
                # 这不会影响速度，因为这一步是存储，而下一步 Global Score 计算时会移回 GPU。
                results.append(SegmentArtifact(
                    text=text,
                    H_k=H_k.cpu(), 
                    e_tilde_k=e_tilde_k.cpu(),
                    f_norm=f_norm,
                    segment_score=f_seg,
                    token_scores=token_scores
                ))
                
            return results

    def compute_chain_global_score(self, chain: List[SegmentArtifact]) -> Tuple[float, float, float]:
        """
        [Section 3.4] 全局评分 (GPU 计算)
        """
        if not chain:
            return 0.0, 0.0, 0.0
            
        K = len(chain)
        
        # 1. Fact Score (CPU scalar calc is fast enough)
        numerators = [seg.f_norm * seg.evidence_score for seg in chain]
        denom = sum(numerators) + 1e-8
        f_fact = 0.0
        for k in range(K):
            w_k = numerators[k] / denom
            f_fact += w_k * chain[k].segment_score
            
        # 2. Logical Coherence (GPU Vector calc)
        if K < 2:
            f_logic = 1.0
        else:
            logic_scores = []
            for k in range(K - 1):
                seg_curr = chain[k]
                seg_next = chain[k+1]
                
                # [GPU] Move vectors back to GPU for fast calculation
                H_k = seg_curr.H_k.to(self.device)
                H_next = seg_next.H_k.to(self.device)
                e_k = seg_curr.e_tilde_k.to(self.device)
                e_next = seg_next.e_tilde_k.to(self.device)
                
                # GPU Cosine
                cos_hidden = F.cosine_similarity(H_k, H_next, dim=0).item()
                cos_embed = F.cosine_similarity(e_k, e_next, dim=0).item()
                
                sim_ctx = (1.0 + cos_embed) / 2.0
                logic_scores.append(sim_ctx * cos_hidden)
                
            f_logic = sum(logic_scores) / (K - 1)
            
        # 3. Global Score
        denominator = f_fact + f_logic - (f_fact * f_logic)
        if abs(denominator) < 1e-6:
            f_global = 0.0
        else:
            f_global = (f_fact * f_logic) / denominator
            
        return f_global, f_fact, f_logic



    def check_refinement(self, artifact: SegmentArtifact) -> RefinementAdvice:
        """
        [Section 3.3] Determine if local refinement is needed based on thresholds.
        """
        # Apply dynamic thresholds (Eq. 15 context)
        adj_low = self.config.tau_seg_low - self.current_threshold_adj
        adj_high = self.config.tau_seg_high + self.current_threshold_adj
        
        if adj_low <= artifact.segment_score < adj_high:
            # Locate lowest scoring token a_low
            # Handle empty token scores gracefully
            if not artifact.token_scores:
                return RefinementAdvice(False)

            min_idx = np.argmin(artifact.token_scores)
            # Window W_k definition
            start = max(0, min_idx - 1)
            end = min(len(artifact.token_scores), min_idx + 2)
            return RefinementAdvice(True, (start, end), min_idx)
            
        return RefinementAdvice(False)
    
    # =========================================================
    # [新增] 严格复现 Eq. 10: Segment-Level Local Refinement
    # =========================================================
    def refine_segment(self, context_text: str, artifact: SegmentArtifact, h_x: torch.Tensor, max_retries: int = 3) -> SegmentArtifact:
        """
        [Token-Guard v2.0] Active Refinement with Self-Correction Prompts.
        不再盲目重写，而是注入“反思引导词”来强迫模型修正幻觉。
        """
        advice = self.check_refinement(artifact)
        if not advice.needed or max_retries <= 0:
            return artifact

        # print(f"  [Refine] Triggered on: '{artifact.text[:30]}...'")
        
        best_artifact = artifact
        
        # 定义反思引导词 (Self-Correction Triggers)
        # 这些词会插入到截断处，诱导模型进入“纠错模式”
        correction_triggers = [
            " Wait, let me double check the details.", # 温和反思
            " Actually, looking closer at the context,", # 深度反思
            " To be precise," # 事实精确化
        ]

        for i in range(max_retries):
            tokens_scores = current_artifact.token_scores
            if not tokens_scores: break
            
            # 1. 定位弱点
            min_idx = np.argmin(tokens_scores)
            
            # 2. 截断 (Context Cutoff)
            cutoff_idx = max(0, min_idx - 2) 
            seg_input_ids = self.tokenizer.encode(current_artifact.text, add_special_tokens=False)
            if cutoff_idx >= len(seg_input_ids): cutoff_idx = max(0, len(seg_input_ids) - 1)

            prefix_ids = seg_input_ids[:cutoff_idx]
            prefix_text = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
            
            # 3. [关键修改] 注入反思引导 (Prompt Injection)
            # 轮询使用不同的引导词
            trigger = correction_triggers[i % len(correction_triggers)]
            
            # 构造新的输入： 原文Context + 当前前缀 + 引导词
            # 模型会接着引导词继续写，往往能修正之前的错误
            guided_input_text = context_text + prefix_text + trigger
            
            inputs = self.tokenizer(
                guided_input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4096
            ).to(self.device)
            
            # 4. 生成修正后的后缀
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=len(seg_input_ids) - cutoff_idx + 30, # 多留点空间给修正
                    do_sample=True,
                    temperature=0.7, # 稍微降低温度，因为引导词已经提供了扰动，需要模型聚焦
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 5. 清洗输出 (去除引导词，只保留修正后的事实)
            # outputs 包含了 Context + Prefix + Trigger + Suffix
            # 我们需要提取 Suffix
            full_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 这是一个简单的字符串处理，提取 trigger 之后的内容
            # 实际情况可能需要更复杂的解析，这里简化处理
            if trigger in full_decoded:
                new_suffix = full_decoded.split(trigger)[-1].strip()
            else:
                # Fallback: 按长度截取
                input_len = inputs.input_ids.shape[1]
                new_suffix = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            # 组合成新片段 (去掉 trigger，让文本看起来自然)
            # 注意：有时候保留 Trigger 会导致句子不通顺，我们这里尝试只保留内容
            # 或者，我们可以认为 new_segment = prefix + new_suffix (这可能包含修正后的事实)
            new_segment_text = prefix_text + " " + new_suffix
            
            # 6. 验证
            refined_artifacts = self.verify_candidates(context_text, [new_segment_text], h_x)
            if not refined_artifacts: break
            refined_artifact = refined_artifacts[0]
            
            # 7. 决策
            # print(f"    Iter {i+1} [Trigger: '{trigger.strip()}']: Score {refined_artifact.segment_score:.3f} (Old: {current_artifact.segment_score:.3f})")
            # print(f"      -> New Text: {new_segment_text[:50]}...")

            if refined_artifact.segment_score > best_artifact.segment_score:
                best_artifact = refined_artifact
                current_artifact = refined_artifact
                
                # 动态阈值检查
                adj_high = self.config.tau_seg_high + self.current_threshold_adj
                if best_artifact.segment_score >= adj_high:
                    break

        return best_artifact


    def adjust_thresholds(self, f_fact: float, f_logic: float):
        """
        [Eq. 15] 动态阈值调整
        """
        if f_fact < 0.5 and f_logic > 0.6: 
            self.current_threshold_adj = self.config.delta_tau
            print("[TokenGuard] Thresholds TIGHTENED (Low Factuality).")
        elif f_logic < 0.5 and f_fact > 0.6:
            self.current_threshold_adj = -self.config.delta_tau
            print("[TokenGuard] Thresholds RELAXED (Low Logic).")
        else:
            self.current_threshold_adj = 0.0