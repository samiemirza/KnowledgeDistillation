import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import get_module, count_params


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    
    Args:
        named_layers (list of tuples): List of tuples in the form (layer_name, layer_module)
    """
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []
        
        def hook_fn(name):
            def _hook(module, input, output):
                # Handle tuple outputs (common in transformers)
                if isinstance(output, tuple):
                    self.features[name] = output[0]  # Usually hidden states
                else:
                    self.features[name] = output
            return _hook
        
        # Register a forward hook for each named layer.
        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    def clear(self):
        """Clears the stored features."""
        self.features.clear()
        
    def remove(self):
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class HintLoss(nn.Module):
    """
    Computes an MSE loss between the teacher's features and the student's features,
    with optional adaptation of the student's features to match the teacher's dimensions.
    
    For transformers, features are typically of shape (batch_size, seq_len, hidden_dim).
    
    Args:
        teacher_channels (int): Hidden dimension in the teacher's feature representation.
        student_channels (int): Hidden dimension in the student's feature representation.
        adapter (str): Specifies whether to attach the adapter on the student or the teacher.
        pooling (str): Pooling strategy for sequence dimension ('none', 'mean', 'cls').
                      - 'none': Match all tokens
                      - 'mean': Average pool over sequence dimension
                      - 'cls': Use only the [CLS] token (first token)
    """
    def __init__(self, teacher_channels, student_channels, adapter, pooling='none'):
        super(HintLoss, self).__init__()

        self.adapter = adapter
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels
        self.pooling = pooling

        if teacher_channels == student_channels:
            self.adaptation = nn.Identity()
        elif adapter == 'student':
            # Linear projection from student dimension to teacher dimension
            self.adaptation = nn.Linear(
                student_channels, 
                teacher_channels,
                bias=False
            ) 
        elif adapter == 'teacher':
            # Linear projection from teacher dimension to student dimension
            self.adaptation = nn.Linear(
                teacher_channels,
                student_channels, 
                bias=False
            ) 
        else:
            raise ValueError(f'{adapter} is not valid. Choose from student or teacher.')

        if not isinstance(self.adaptation, nn.Identity):
            print(f'{adapter} adapter has {count_params(self.adaptation)} params...\n')

    def _apply_pooling(self, features):
        """
        Apply pooling strategy to sequence features.
        
        Args:
            features (Tensor): Shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Tensor: Pooled features
        """
        if self.pooling == 'mean':
            return features.mean(dim=1)  # (batch_size, hidden_dim)
        elif self.pooling == 'cls':
            return features[:, 0, :]  # (batch_size, hidden_dim)
        else:  # 'none'
            return features

    def _align_sequence_length(self, source, target):
        """
        Align sequence lengths between source and target features.
        If lengths differ, truncate or pad the source to match target.
        
        Args:
            source (Tensor): Shape (batch_size, seq_len_source, hidden_dim)
            target (Tensor): Shape (batch_size, seq_len_target, hidden_dim)
            
        Returns:
            Tensor: Source with aligned sequence length
        """
        if source.shape[1] == target.shape[1]:
            return source
        
        seq_len_source = source.shape[1]
        seq_len_target = target.shape[1]
        
        if seq_len_source > seq_len_target:
            # Truncate
            return source[:, :seq_len_target, :]
        else:
            # Pad with zeros
            padding = torch.zeros(
                source.shape[0], 
                seq_len_target - seq_len_source, 
                source.shape[2],
                device=source.device,
                dtype=source.dtype
            )
            return torch.cat([source, padding], dim=1)

    def forward(self, teacher_features, student_features):
        """
        Compute hint loss between teacher and student features.
        
        Args:
            teacher_features (Tensor): Shape (batch_size, seq_len, teacher_channels)
            student_features (Tensor): Shape (batch_size, seq_len, student_channels)
            
        Returns:
            Tensor: Scalar loss value
        """
        # Adapt student's features
        if self.adapter == 'student':
            adapted_student = self.adaptation(student_features)
            
            # Align sequence lengths if necessary
            if adapted_student.shape[1] != teacher_features.shape[1]:
                adapted_student = self._align_sequence_length(adapted_student, teacher_features)
            
            # Apply pooling if specified
            adapted_student = self._apply_pooling(adapted_student)
            teacher_features = self._apply_pooling(teacher_features)
            
            return F.mse_loss(adapted_student, teacher_features)
        
        # Adapt teacher's features
        elif self.adapter == 'teacher':
            adapted_teacher = self.adaptation(teacher_features)
            
            # Align sequence lengths if necessary
            if adapted_teacher.shape[1] != student_features.shape[1]:
                adapted_teacher = self._align_sequence_length(adapted_teacher, student_features)
            
            # Apply pooling if specified
            adapted_teacher = self._apply_pooling(adapted_teacher)
            student_features = self._apply_pooling(student_features)
            
            return F.mse_loss(adapted_teacher, student_features)


class FitNets(nn.Module):
    """
    FitNets for knowledge distillation with LLMs/Transformers. This module trains a 
    student network using hints from intermediate hidden states of a teacher network.
    
    Args:
        teacher (nn.Module): Pretrained teacher network (e.g., transformer model).
        student (nn.Module): Student network to be trained.
        teacher_layer (str): Teacher layer name for feature extraction.
                            For HuggingFace models, use format like 'transformer.h.6' 
                            or 'model.layers.6'
        student_layer (str): Student layer name for feature extraction.
        teacher_channels (int): Hidden dimension in teacher feature map.
        student_channels (int): Hidden dimension in student feature map.
        adapter (str): Type of adapter ('student' or 'teacher').
        pooling (str): Pooling strategy ('none', 'mean', 'cls').
    """
    def __init__(self, teacher, student, teacher_layer, student_layer, 
                 teacher_channels, student_channels, adapter='student', pooling='none'):
        super(FitNets, self).__init__()
        self.teacher = teacher
        self.student = student
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.teacher_layer = teacher_layer
        self.student_layer = student_layer
        
        # Register hooks to capture intermediate features
        try:
            self.teacher_hooks = FeatureHooks([
                (teacher_layer, get_module(self.teacher, teacher_layer))
            ])
            self.student_hooks = FeatureHooks([
                (student_layer, get_module(self.student, student_layer))
            ])
        except AttributeError as e:
            raise ValueError(
                f"Could not find layer. Make sure layer names are correct.\n"
                f"Teacher layer: {teacher_layer}\n"
                f"Student layer: {student_layer}\n"
                f"Error: {e}"
            )
        
        # Create hint loss criterion
        self.hint_criterion = HintLoss(
            teacher_channels, 
            student_channels, 
            adapter,
            pooling=pooling
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass that computes both teacher and student outputs and hint loss.
        
        Args:
            input_ids (Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, seq_len).
            **kwargs: Additional arguments to pass to the models.
            
        Returns:
            tuple: (teacher_logits, student_logits, hint_loss)
        """
        # Prepare inputs
        teacher_inputs = {'input_ids': input_ids}
        student_inputs = {'input_ids': input_ids}
        
        if attention_mask is not None:
            teacher_inputs['attention_mask'] = attention_mask
            student_inputs['attention_mask'] = attention_mask
        
        # Add any additional kwargs
        teacher_inputs.update(kwargs)
        student_inputs.update(kwargs)
        
        # Forward pass through teacher and student networks
        with torch.no_grad():
            teacher_output = self.teacher(**teacher_inputs)
        student_output = self.student(**student_inputs)
        
        # Extract logits (handle different output formats)
        if hasattr(teacher_output, 'logits'):
            teacher_logits = teacher_output.logits
        elif isinstance(teacher_output, tuple):
            teacher_logits = teacher_output[0]
        else:
            teacher_logits = teacher_output
            
        if hasattr(student_output, 'logits'):
            student_logits = student_output.logits
        elif isinstance(student_output, tuple):
            student_logits = student_output[0]
        else:
            student_logits = student_output

        # Extract intermediate features
        teacher_feature = self.teacher_hooks.features.get(self.teacher_layer)
        student_feature = self.student_hooks.features.get(self.student_layer)

        if teacher_feature is None or student_feature is None:
            raise ValueError(
                f"Missing features for layers.\n"
                f"Teacher layer '{self.teacher_layer}': {teacher_feature is not None}\n"
                f"Student layer '{self.student_layer}': {student_feature is not None}\n"
                f"Available teacher features: {list(self.teacher_hooks.features.keys())}\n"
                f"Available student features: {list(self.student_hooks.features.keys())}"
            )
        
        # Compute hint loss
        hint_loss = self.hint_criterion(teacher_feature, student_feature)
        
        # Clear features for next forward pass
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        return teacher_logits, student_logits, hint_loss
    
    def cleanup(self):
        """Remove all hooks. Call this when done training."""
        self.teacher_hooks.remove()
        self.student_hooks.remove()