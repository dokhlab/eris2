import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_proj(out)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        attn_output = self.mha(self.layer_norm1(x1))
        x1 = x1 + self.dropout(attn_output)
        attn_output = self.mha(self.layer_norm2(x2))
        x2 = x2 + self.dropout(attn_output)
        return x1, x2

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(EnhancedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return F.relu(out)

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)

class DDGPredictor(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads=8):
        super(DDGPredictor, self).__init__()
        self.embedding_dim = embedding_dim
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Feature attention and fusion
        self.feature_attention = FeatureAttention(512)
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Enhanced embedding processing
        self.embedding_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Multi-head attention layers
        self.self_attention = nn.ModuleList([
            MultiHeadAttention(embedding_dim, num_attention_heads)
            for _ in range(3)
        ])
        
        # Cross-attention
        self.cross_attention = CrossAttention(embedding_dim, num_attention_heads)
        
        # Distance matrix processors
        self.ca_conv_layers = nn.Sequential(
            EnhancedResidualBlock(1, 32),
            EnhancedResidualBlock(32, 64),
            EnhancedResidualBlock(64, 128),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.atom_dist_conv_layers = nn.Sequential(
            EnhancedResidualBlock(1, 32),
            EnhancedResidualBlock(32, 64),
            EnhancedResidualBlock(64, 128),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature processors
        self.rsa_processor = nn.Sequential(
            EnhancedResidualBlock(1, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.angles_processor = nn.Sequential(
            EnhancedResidualBlock(2, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.hbond_processor = nn.Sequential(
            EnhancedResidualBlock(3, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.ss_processor = nn.Sequential(
            EnhancedResidualBlock(8, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.charge_processor = nn.Sequential(
            EnhancedResidualBlock(1, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.hydrophobicity_processor = nn.Sequential(
            EnhancedResidualBlock(1, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.atom_processor = nn.Sequential(
            EnhancedResidualBlock(4, 16),
            EnhancedResidualBlock(16, 32),
            EnhancedResidualBlock(32, 64),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(64)
        
        # Calculate input dimension for FC layers
        fc_input_dim = (128 + # ca distance features
                       128 + # atom distance features
                       embedding_dim * 2 + # wt and mut embeddings
                       64 + # rsa features
                       64 + # backbone angles features
                       64 + # hydrogen bond features
                       64 + # secondary structure features
                       64 + # charge features
                       64 + # hydrophobicity features
                       64) # atom type features
        
        # FC layers with residual connections
        self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fc_input_dim if i == 0 else 512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for i in range(3)
        ])
        
        self.final_layer = nn.Linear(512, 1)
        
    def forward(self, wt_embedding, mut_embedding, ca_distance_matrix, atom_distance_matrix,
                rsa_values, backbone_angles, hbond_features, ss_features,
                charge_features, hydrophobicity_features, atom_features):
        
        # Process embeddings
        wt_processed = self.embedding_processor(wt_embedding)
        mut_processed = self.embedding_processor(mut_embedding)
        
        # Apply self-attention
        for attention_layer in self.self_attention:
            wt_attn = self.layer_norm(wt_processed + attention_layer(wt_processed))
            mut_attn = self.layer_norm(mut_processed + attention_layer(mut_processed))
        
        # Cross-attention
        wt_attn, mut_attn = self.cross_attention(wt_attn, mut_attn)
        
        # Process distance matrices
        ca_dist = ca_distance_matrix.unsqueeze(1)
        ca_dist = ca_dist.reshape(ca_dist.size(0), 1, -1)
        ca_conv_features = self.ca_conv_layers(ca_dist).squeeze(-1)
        
        atom_dist = atom_distance_matrix.unsqueeze(1)
        atom_dist = atom_dist.reshape(atom_dist.size(0), 1, -1)
        atom_dist_features = self.atom_dist_conv_layers(atom_dist).squeeze(-1)
        
        # Process other features
        rsa = rsa_values.unsqueeze(1)
        rsa_features = self.feature_norm(self.rsa_processor(rsa).squeeze(-1))
        
        angles = backbone_angles.transpose(1, 2)
        angles_features = self.feature_norm(self.angles_processor(angles).squeeze(-1))
        
        hbonds = hbond_features.transpose(1, 2)
        hbond_features = self.feature_norm(self.hbond_processor(hbonds).squeeze(-1))
        
        ss = ss_features.transpose(1, 2)
        ss_features = self.feature_norm(self.ss_processor(ss).squeeze(-1))
        
        charge = charge_features.transpose(1, 2)
        charge_features = self.feature_norm(self.charge_processor(charge).squeeze(-1))
        
        hydrophobicity = hydrophobicity_features.transpose(1, 2)
        hydrophobicity_features = self.feature_norm(self.hydrophobicity_processor(hydrophobicity).squeeze(-1))
        
        atoms = atom_features.transpose(1, 2)
        atom_features = self.feature_norm(self.atom_processor(atoms).squeeze(-1))
        
        # Reshape attention outputs
        wt_attn = wt_attn.mean(dim=1)
        mut_attn = mut_attn.mean(dim=1)
        
        # Combine all features
        combined = torch.cat((
            ca_conv_features,
            atom_dist_features,
            wt_attn,
            mut_attn,
            rsa_features,
            angles_features,
            hbond_features,
            ss_features,
            charge_features,
            hydrophobicity_features,
            atom_features
        ), dim=1)
        
        # Process through FC layers with residual connections
        x = combined
        for fc_layer in self.fc_layers:
            x_residual = x
            x = fc_layer(x)
            if x.size() == x_residual.size():
                x = x + x_residual
        
        # Final prediction
        output = self.final_layer(x)
        return output.squeeze(-1)
