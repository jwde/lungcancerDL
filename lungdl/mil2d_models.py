import torch 
from cnn_to_grayscale import cnn_to_bw                                                 
import math  
from xgboost_slicer import PretrainedFeaturesXGBoost                                   

def xavier_init2d(modules):                                                            
     for m in modules:
         if isinstance(m, nn.Conv2d):                                                   
             n = m.kernel_size[0] * m.kernel_size[1]                                    
             n *= m.in_channels                                                         
             var = math.sqrt(2./n)                                                      
             m.weight.data.normal_(0,var)                                               
 
 class PretrainedMIL(nn.Module):                                                        
     def __init__(self, features, mil_Scores):                                          
         super(PretrainedMIL, self).__init__()                                          
         self.features = features # a pretrained features extractor                     
         self.mil_scores = mil_scores                                                   
         xavier_init2d(self.mil_scores.modules())                                       
 
     def forward(self, xs):
 
         # Forward pass feature extractor to extract *Instances*                        
         xs.volatile = True                                                             
         feats = self.features(xs)
         feats.volatile = False
         feats.requires_grad = True                                                     
 
         # Forward pass instance scorer and make prediction
         scores = self.mil_scores(feats).view(N,-1)
         probs = nn.functional.sigmoid(scores.max(1)[0].view(N,-1))                     
         return probs, scores
