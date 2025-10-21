

class DualLearningLoss(nn.Module):
    """
    Dual learning loss combining image reconstruction and CRF prediction.

    Key improvements:
    - Normalized CRF values [0, 1] for better gradient balance
    - Charbonnier loss (smooth L1) for reconstruction - more robust than L1/L2
    - Optional perceptual loss with LPIPS
    - Adaptive weighting based on loss magnitudes
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        crf_weight: float = 0.1,
        use_perceptual: bool = False,
        perceptual_weight: float = 1.0,
        crf_min: float = 23.0,
        crf_max: float = 63.0,
        charbonnier_eps: float = 1e-3
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.crf_weight = crf_weight
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight

        # CRF normalization parameters
        self.crf_min = crf_min
        self.crf_max = crf_max
        self.crf_range = crf_max - crf_min

        # Charbonnier loss epsilon (smooth L1)
        self.eps = charbonnier_eps

        # LPIPS will be initialized on first forward pass (device-aware)
        self.perceptual_loss = None
        if use_perceptual and not LPIPS_AVAILABLE:
            logger.warning("LPIPS requested but not installed. Using Charbonnier only.")
            self.use_perceptual = False

        logger.info(
            f"DualLearningLoss: λ_rec={reconstruction_weight}, λ_crf={crf_weight}, "
            f"perceptual={use_perceptual}, CRF range=[{crf_min}, {crf_max}]"
        )

    def normalize_crf(self, crf: torch.Tensor) -> torch.Tensor:
        """Normalize CRF values from [crf_min, crf_max] to [0, 1]."""
        return (crf - self.crf_min) / self.crf_range

    def denormalize_crf(self, crf_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize CRF values from [0, 1] to [crf_min, crf_max]."""
        return crf_norm * self.crf_range + self.crf_min

    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Charbonnier loss (smooth L1 variant).
        More robust to outliers than L2, smoother than L1.
        sqrt((pred - target)^2 + eps^2)
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()

    def forward(
        self,
        reconstructed_image: torch.Tensor,
        target_image: torch.Tensor,
        predicted_crf: torch.Tensor,
        target_crf: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual learning loss.

        Args:
            reconstructed_image: Predicted HQ image [B, C, H, W], range [0, 1] or [-1, 1]
            target_image: Ground truth HQ image [B, C, H, W]
            predicted_crf: Predicted CRF values [B, 1] or [B], UNNORMALIZED (23-63)
            target_crf: Ground truth CRF values [B, 1] or [B], UNNORMALIZED (23-63)

        Returns:
            Dictionary of loss components
        """

        # ===== 1. Implicit Learning: Image Reconstruction =====

        # Primary reconstruction loss (Charbonnier)
        reconstruction_loss = self.charbonnier_loss(reconstructed_image, target_image)

        # Optional perceptual loss
        if self.use_perceptual:
            # Lazy initialization of LPIPS on correct device
            if self.perceptual_loss is None:
                self.perceptual_loss = lpips.LPIPS(net='alex').to(reconstructed_image.device)
                self.perceptual_loss.eval()
                for param in self.perceptual_loss.parameters():
                    param.requires_grad = False

            # Ensure images are in [-1, 1] range for LPIPS
            if reconstructed_image.min() >= 0:  # If in [0, 1]
                recon_norm = reconstructed_image * 2 - 1
                target_norm = target_image * 2 - 1
            else:  # Already in [-1, 1]
                recon_norm = reconstructed_image
                target_norm = target_image

            perceptual_loss = self.perceptual_loss(recon_norm, target_norm).mean()

            # Combine with perceptual
            total_reconstruction_loss = reconstruction_loss + self.perceptual_weight * perceptual_loss
        else:
            perceptual_loss = torch.tensor(0.0, device=reconstructed_image.device)
            total_reconstruction_loss = reconstruction_loss

        # ===== 2. Explicit Learning: CRF Prediction =====

        # Normalize CRF values to [0, 1] for balanced gradients
        predicted_crf_norm = self.normalize_crf(predicted_crf.squeeze())
        target_crf_norm = self.normalize_crf(target_crf.squeeze())

        # Use smooth L1 (Huber-like) for CRF prediction
        # This is more robust than MSE for regression
        crf_loss = F.smooth_l1_loss(predicted_crf_norm, target_crf_norm, beta=0.1)

        # ===== 3. Weighted Total Loss =====

        reconstruction_loss_weighted = self.reconstruction_weight * total_reconstruction_loss
        crf_loss_weighted = self.crf_weight * crf_loss
        total_loss = reconstruction_loss_weighted + crf_loss_weighted

        # ===== 4. Return all components for logging =====

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,  # Charbonnier only
            'perceptual_loss': perceptual_loss,
            'crf_loss': crf_loss,
            'reconstruction_loss_weighted': reconstruction_loss_weighted,
            'crf_loss_weighted': crf_loss_weighted,
            # Denormalized CRF for interpretability
            'predicted_crf_raw': self.denormalize_crf(predicted_crf_norm).mean(),
            'target_crf_raw': target_crf.mean()
        }


class AdaptiveDualLearningLoss(DualLearningLoss):
    """
    Adaptive version that automatically balances reconstruction and CRF losses
    based on their relative magnitudes (uncertainty weighting approach).
    """

    def __init__(
        self,
        initial_log_var_rec: float = 0.0,
        initial_log_var_crf: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Learnable uncertainty parameters (log variance)
        # Using log(σ²) for numerical stability
        self.log_var_rec = nn.Parameter(torch.tensor(initial_log_var_rec))
        self.log_var_crf = nn.Parameter(torch.tensor(initial_log_var_crf))

        logger.info("Using AdaptiveDualLearningLoss with learnable task weighting")

    def forward(
        self,
        reconstructed_image: torch.Tensor,
        target_image: torch.Tensor,
        predicted_crf: torch.Tensor,
        target_crf: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive dual learning loss with uncertainty weighting.
        Loss = (1/σ²_rec)*L_rec + (1/σ²_crf)*L_crf + log(σ²_rec) + log(σ²_crf)
        """

        # Get base losses
        base_losses = super().forward(
            reconstructed_image, target_image, predicted_crf, target_crf
        )

        # Extract unweighted losses
        rec_loss = base_losses['reconstruction_loss']
        if self.use_perceptual:
            rec_loss = rec_loss + self.perceptual_weight * base_losses['perceptual_loss']
        crf_loss = base_losses['crf_loss']

        # Adaptive weighting using learned uncertainties
        # precision = exp(-log_var) = 1/σ²
        precision_rec = torch.exp(-self.log_var_rec)
        precision_crf = torch.exp(-self.log_var_crf)

        # Weighted losses with regularization
        weighted_rec = precision_rec * rec_loss + self.log_var_rec
        weighted_crf = precision_crf * crf_loss + self.log_var_crf

        total_loss = weighted_rec + weighted_crf

        # Update returned dictionary
        base_losses.update({
            'total_loss': total_loss,
            'reconstruction_loss_weighted': weighted_rec,
            'crf_loss_weighted': weighted_crf,
            'uncertainty_rec': torch.exp(self.log_var_rec * 0.5),  # σ
            'uncertainty_crf': torch.exp(self.log_var_crf * 0.5),
            'weight_rec': precision_rec,
            'weight_crf': precision_crf
        })

        return base_losses


class BaselineRelativeLoss(nn.Module):
    """
    Baseline-Relative Dual Learning Loss.
    
    The model must BEAT the baseline (doing nothing) to get credit.
    
    Key concepts:
    1. Reconstruction baseline: L_baseline_rec = loss(LQ_image, HQ_image)
       - The "do nothing" baseline just outputs the LQ image as-is
    2. Model improvement: Δ_rec = L_baseline_rec - L_model_rec
       - Positive Δ = model is better than baseline (reward)
       - Negative Δ = model is worse than baseline (penalty)
    3. Only reward improvements: max(0, Δ_rec) or use soft version
    
    For CRF prediction:
    - Baseline: Always predict the mean CRF (random guess)
    - Model must beat this naive baseline
    
    Benefits:
    - Prevents trivial solutions (e.g., just copy input)
    - Focuses learning on actual improvements
    - More interpretable: loss shows improvement margin
    - Naturally prevents overfitting to "safe" solutions
    
    Args:
        reconstruction_weight: Weight for reconstruction improvement
        crf_weight: Weight for CRF prediction improvement
        use_perceptual: Add perceptual loss
        margin: Margin for improvement (model must beat baseline by this much)
        soft_margin: Use smooth relu instead of hard threshold
        crf_min: Minimum CRF value for normalization
        crf_max: Maximum CRF value for normalization
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        crf_weight: float = 0.1,
        use_perceptual: bool = False,
        perceptual_weight: float = 1.0,
        margin: float = 0.0,
        soft_margin: bool = True,
        crf_min: float = 23.0,
        crf_max: float = 63.0,
        charbonnier_eps: float = 1e-3
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.crf_weight = crf_weight
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        self.margin = margin
        self.soft_margin = soft_margin
        
        # CRF normalization
        self.crf_min = crf_min
        self.crf_max = crf_max
        self.crf_range = crf_max - crf_min
        
        # Charbonnier loss epsilon
        self.eps = charbonnier_eps
        
        # LPIPS (lazy initialization)
        self.perceptual_loss = None
        if use_perceptual and not LPIPS_AVAILABLE:
            logger.warning("LPIPS not available. Disabling perceptual loss.")
            self.use_perceptual = False
        
        # Running statistics for CRF baseline
        self.register_buffer('crf_mean', torch.tensor(0.0))
        self.register_buffer('crf_count', torch.tensor(0))
        
        logger.info(
            f"BaselineRelativeLoss initialized: "
            f"λ_rec={reconstruction_weight}, λ_crf={crf_weight}, "
            f"margin={margin}, soft_margin={soft_margin}"
        )
    
    def charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Charbonnier loss (smooth L1)."""
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()
    
    def normalize_crf(self, crf: torch.Tensor) -> torch.Tensor:
        """Normalize CRF values to [0, 1]."""
        return (crf - self.crf_min) / self.crf_range
    
    def update_crf_baseline(self, target_crf: torch.Tensor):
        """Update running mean of CRF values for baseline."""
        batch_mean = target_crf.mean()
        batch_size = target_crf.numel()
        
        # Exponential moving average
        alpha = batch_size / (self.crf_count + batch_size)
        self.crf_mean = (1 - alpha) * self.crf_mean + alpha * batch_mean
        self.crf_count += batch_size
    
    def forward(
        self,
        reconstructed_image: torch.Tensor,
        lq_image: torch.Tensor,  # ADDED: Need this for baseline
        target_image: torch.Tensor,
        predicted_crf: torch.Tensor,
        target_crf: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute baseline-relative loss.
        
        Args:
            reconstructed_image: Model's reconstruction [B, C, H, W]
            lq_image: Low-quality input [B, C, H, W] (the "do nothing" baseline)
            target_image: Ground truth HQ image [B, C, H, W]
            predicted_crf: Model's CRF prediction [B]
            target_crf: Ground truth CRF [B]
        
        Returns:
            Dictionary of loss components
        """
        
        # ===== 1. RECONSTRUCTION: Beat the "do nothing" baseline =====
        
        # Baseline loss: What if we just output the LQ image?
        baseline_recon_loss = self.charbonnier_loss(lq_image, target_image)
        
        # Model loss: How well does our reconstruction do?
        model_recon_loss = self.charbonnier_loss(reconstructed_image, target_image)
        
        # Improvement over baseline (positive = good, negative = bad)
        recon_improvement = baseline_recon_loss - model_recon_loss
        
        # Only reward improvements beyond margin
        if self.soft_margin:
            # Smooth version: penalize if worse, reward if better than margin
            recon_loss_relative = -F.relu(recon_improvement - self.margin) + model_recon_loss
        else:
            # Hard version: only count improvements
            recon_loss_relative = torch.where(
                recon_improvement > self.margin,
                -recon_improvement + model_recon_loss,
                model_recon_loss + self.margin  # Penalty for not improving
            )
        
        # Perceptual loss (also baseline-relative)
        if self.use_perceptual:
            if self.perceptual_loss is None:
                self.perceptual_loss = lpips.LPIPS(net='alex').to(reconstructed_image.device)
                self.perceptual_loss.eval()
                for param in self.perceptual_loss.parameters():
                    param.requires_grad = False
            
            # Normalize to [-1, 1] for LPIPS
            if reconstructed_image.min() >= 0:
                recon_norm = reconstructed_image * 2 - 1
                target_norm = target_image * 2 - 1
                lq_norm = lq_image * 2 - 1
            else:
                recon_norm = reconstructed_image
                target_norm = target_image
                lq_norm = lq_image
            
            baseline_perceptual = self.perceptual_loss(lq_norm, target_norm).mean()
            model_perceptual = self.perceptual_loss(recon_norm, target_norm).mean()
            
            perceptual_improvement = baseline_perceptual - model_perceptual
            perceptual_loss_relative = -F.relu(perceptual_improvement - self.margin) + model_perceptual
            
            total_recon_loss = recon_loss_relative + self.perceptual_weight * perceptual_loss_relative
        else:
            perceptual_improvement = torch.tensor(0.0, device=reconstructed_image.device)
            baseline_perceptual = torch.tensor(0.0, device=reconstructed_image.device)
            model_perceptual = torch.tensor(0.0, device=reconstructed_image.device)
            total_recon_loss = recon_loss_relative
        
        # ===== 2. CRF PREDICTION: Beat the "always guess mean" baseline =====
        
        # Update baseline (running mean of target CRFs)
        if self.training:
            self.update_crf_baseline(target_crf.detach())
        
        # Normalize CRFs
        predicted_crf_norm = self.normalize_crf(predicted_crf.squeeze())
        target_crf_norm = self.normalize_crf(target_crf.squeeze())
        baseline_crf_norm = self.normalize_crf(self.crf_mean)
        
        # Baseline loss: Always predict the mean CRF
        baseline_crf_loss = F.smooth_l1_loss(
            baseline_crf_norm.expand_as(target_crf_norm), 
            target_crf_norm, 
            beta=0.1
        )
        
        # Model loss: How well does our prediction do?
        model_crf_loss = F.smooth_l1_loss(predicted_crf_norm, target_crf_norm, beta=0.1)
        
        # Improvement over baseline
        crf_improvement = baseline_crf_loss - model_crf_loss
        
        # Only reward improvements
        if self.soft_margin:
            crf_loss_relative = -F.relu(crf_improvement - self.margin) + model_crf_loss
        else:
            crf_loss_relative = torch.where(
                crf_improvement > self.margin,
                -crf_improvement + model_crf_loss,
                model_crf_loss + self.margin
            )
        
        # ===== 3. TOTAL LOSS =====
        
        reconstruction_loss_weighted = self.reconstruction_weight * total_recon_loss
        crf_loss_weighted = self.crf_weight * crf_loss_relative
        total_loss = reconstruction_loss_weighted + crf_loss_weighted
        
        # ===== 4. RETURN DETAILED METRICS =====
        
        return {
            # Main losses
            'total_loss': total_loss,
            'reconstruction_loss': model_recon_loss,  # Absolute model loss
            'reconstruction_loss_relative': recon_loss_relative,  # Relative to baseline
            'crf_loss': model_crf_loss,  # Absolute model loss
            'crf_loss_relative': crf_loss_relative,  # Relative to baseline
            'reconstruction_loss_weighted': reconstruction_loss_weighted,
            'crf_loss_weighted': crf_loss_weighted,
            
            # Baselines
            'baseline_reconstruction_loss': baseline_recon_loss,
            'baseline_crf_loss': baseline_crf_loss,
            'crf_baseline_value': self.crf_mean,
            
            # Improvements (positive = good)
            'reconstruction_improvement': recon_improvement,
            'crf_improvement': crf_improvement,
            
            # Perceptual
            'perceptual_loss': model_perceptual,
            'baseline_perceptual_loss': baseline_perceptual,
            'perceptual_improvement': perceptual_improvement,
            
            # CRF values for logging
            'predicted_crf_raw': predicted_crf.mean(),
            'target_crf_raw': target_crf.mean()
        }


class AdaptiveBaselineRelativeLoss(BaselineRelativeLoss):
    """
    Adaptive version with learnable task weights.
    
    Combines baseline-relative learning with adaptive uncertainty weighting.
    Now the model learns:
    1. How much to improve over baseline (baseline-relative)
    2. How confident it is in each task (adaptive weights)
    """
    
    def __init__(
        self,
        initial_log_var_rec: float = 0.0,
        initial_log_var_crf: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Learnable uncertainty parameters
        self.log_var_rec = nn.Parameter(torch.tensor(initial_log_var_rec))
        self.log_var_crf = nn.Parameter(torch.tensor(initial_log_var_crf))
        
        logger.info("Using AdaptiveBaselineRelativeLoss with learnable uncertainty")
    
    def forward(
        self,
        reconstructed_image: torch.Tensor,
        lq_image: torch.Tensor,
        target_image: torch.Tensor,
        predicted_crf: torch.Tensor,
        target_crf: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute adaptive baseline-relative loss."""
        
        # Get base losses
        base_losses = super().forward(
            reconstructed_image, lq_image, target_image,
            predicted_crf, target_crf
        )
        
        # Extract relative losses
        rec_loss = base_losses['reconstruction_loss_relative']
        crf_loss = base_losses['crf_loss_relative']
        
        # Adaptive weighting with uncertainty
        precision_rec = torch.exp(-self.log_var_rec)
        precision_crf = torch.exp(-self.log_var_crf)
        
        weighted_rec = precision_rec * rec_loss + self.log_var_rec
        weighted_crf = precision_crf * crf_loss + self.log_var_crf
        
        total_loss = weighted_rec + weighted_crf
        
        # Update returned losses
        base_losses.update({
            'total_loss': total_loss,
            'reconstruction_loss_weighted': weighted_rec,
            'crf_loss_weighted': weighted_crf,
            'uncertainty_rec': torch.exp(self.log_var_rec * 0.5),
            'uncertainty_crf': torch.exp(self.log_var_crf * 0.5),
            'weight_rec': precision_rec,
            'weight_crf': precision_crf
        })
        
        return base_losses

# Legacy loss for backwards compatibility
class CompressionLoss(nn.Module):
    """
    DEPRECATED: Use DualLearningLoss instead.
    """

    def __init__(self, lambda_rec=1.0):
        super().__init__()
        logger.warning(
            "CompressionLoss is deprecated. Please use DualLearningLoss instead."
        )
        self.ce = nn.CrossEntropyLoss()
        self.lambda_rec = lambda_rec

    def forward(self, crf_pred, crf_true, recon_pred, recon_true):
        l_crf = self.ce(crf_pred, crf_true)
        l_rec = F.l1_loss(recon_pred, recon_true)
        return l_crf + self.lambda_rec * l_rec