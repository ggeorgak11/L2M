
import numpy as np 
import torch
import torch.nn.functional as F


class SemanticGrid(object):
    
    def __init__(self, batch_size, grid_dim, crop_size, cell_size, spatial_labels, object_labels):
        self.grid_dim = grid_dim
        self.cell_size = cell_size
        self.spatial_labels = spatial_labels
        self.object_labels = object_labels
        self.batch_size = batch_size
        self.crop_size = crop_size

        self.crop_start = int( (self.grid_dim[0] / 2) - (self.crop_size / 2) )
        self.crop_end = int( (self.grid_dim[0] / 2) + (self.crop_size / 2) )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # predicted sem grid over entire scene -- initially uniform distribution over the labels
        self.sem_grid = torch.ones((self.batch_size, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.sem_grid = self.sem_grid*(1/self.object_labels)

        # observed ground-projected sem grid over entire scene
        self.proj_grid = torch.ones((self.batch_size, self.spatial_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.proj_grid = self.proj_grid*(1/self.spatial_labels)
        
        # Maps containing accumulated uncertainty
        self.uncertainty_map = torch.zeros((self.batch_size, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        self.per_class_uncertainty_map = torch.zeros((self.batch_size, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)


    # Transform each ground-projected grid into geocentric coordinates
    def spatialTransformer(self, grid, pose, abs_pose):
        # Input: 
        # grid -- sequence len x number of classes x grid_dim x grid_dim
        # pose -- sequence len x 3
        # abs_pose -- same as pose

        geo_grid_out = torch.zeros((grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(grid.device)

        init_pose = abs_pose[0,:] # init absolute pose of each sequence
        init_rot_mat = torch.tensor([[torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                                        [torch.sin(init_pose[2]),torch.cos(init_pose[2])]], dtype=torch.float32).to(grid.device)

        for j in range(grid.shape[0]): # sequence length

            grid_step = grid[j,:,:,:].unsqueeze(0)
            pose_step = pose[j,:]
        
            rel_coord = torch.tensor([pose_step[1],pose_step[0]], dtype=torch.float32).to(grid.device)
            rel_coord = rel_coord.reshape((2,1))
            rel_coord = torch.matmul(init_rot_mat,rel_coord)
    
            x = 2*(rel_coord[0]/self.cell_size)/(self.grid_dim[0])
            z = 2*(rel_coord[1]/self.cell_size)/(self.grid_dim[1])
    
            angle = pose_step[2]

            trans_theta = torch.tensor( [ [1, -0, x], [0, 1, z] ], dtype=torch.float32 ).unsqueeze(0)
            rot_theta = torch.tensor( [ [torch.cos(angle), -1.0*torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0] ], dtype=torch.float32 ).unsqueeze(0)
            trans_theta = trans_theta.to(grid.device)
            rot_theta = rot_theta.to(grid.device)
            
            trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False) # get grid translation displacement
            rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False) # get grid rotation displacement
            
            rot_geo_grid = F.grid_sample(grid_step, rot_disp_grid.float(), align_corners=False ) # apply rotation
            geo_grid = F.grid_sample(rot_geo_grid, trans_disp_grid.float(), align_corners=False) # apply translation

            geo_grid = geo_grid + 1e-12
            geo_grid_out[j,:,:,:] = geo_grid

        return geo_grid_out

    
    # Transform a geocentric map back to egocentric view
    def rotate_map(self, grid, rel_pose, abs_pose):
            # grid -- sequence len x number of classes x grid_dim x grid_dim
            # rel_pose -- sequence len x 3
            ego_grid_out = torch.zeros((grid.shape[0], grid.shape[1], self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(grid.device)
            init_pose = abs_pose[0,:] # init absolute pose of each sequence
            init_rot_mat = torch.tensor([[torch.cos(init_pose[2]), -torch.sin(init_pose[2])],
                                                    [torch.sin(init_pose[2]),torch.cos(init_pose[2])]], dtype=torch.float32).to(grid.device)
            for i in range(grid.shape[0]): # sequence length
                grid_step = grid[i,:,:,:].unsqueeze(0)
                rel_pose_step = rel_pose[i,:]
                rel_coord = torch.tensor([rel_pose_step[1],rel_pose_step[0]], dtype=torch.float32).to(grid.device)
                rel_coord = rel_coord.reshape((2,1))
                rel_coord = torch.matmul(init_rot_mat,rel_coord)
                x = -2*(rel_coord[0]/self.cell_size)/(self.grid_dim[0])
                z = -2*(rel_coord[1]/self.cell_size)/(self.grid_dim[1])
                angle = -rel_pose_step[2]
                
                trans_theta = torch.tensor( [ [1, -0, x], [0, 1, z] ], dtype=torch.float32 ).unsqueeze(0)
                rot_theta = torch.tensor( [ [torch.cos(angle), -1.0*torch.sin(angle), 0], [torch.sin(angle), torch.cos(angle), 0] ], dtype=torch.float32 ).unsqueeze(0)
                trans_theta = trans_theta.to(grid.device)
                rot_theta = rot_theta.to(grid.device)
                
                trans_disp_grid = F.affine_grid(trans_theta, grid_step.size(), align_corners=False) # get grid translation displacement
                rot_disp_grid = F.affine_grid(rot_theta, grid_step.size(), align_corners=False) # get grid rotation displacement
                trans_ego_grid = F.grid_sample(grid_step, trans_disp_grid.float(), align_corners=False ) # apply translation 
                ego_grid = F.grid_sample(trans_ego_grid, rot_disp_grid.float(), align_corners=False) # apply rotation
                ego_grid_out[i,:,:,:] = ego_grid
            return ego_grid_out
    


    def update_sem_grid_bayes(self, geo_grid):
        # Input geo_grid -- B x T x num_of_classes x grid_dim x grid_dim
        # Update the class probabilities at each location of the grid using Bayes rule
        # geo_grid contains the single view observations of the sequence
        step_geo_grid = torch.zeros((geo_grid.shape[0], geo_grid.shape[1], self.object_labels, 
                                                self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(geo_grid.device)
        for i in range(geo_grid.shape[1]): # sequence length
            new_obsv_grid = geo_grid[:,i,:,:,:]
            mul_probs_grid = new_obsv_grid * self.sem_grid
            normalization_grid = torch.sum(mul_probs_grid, dim=1, keepdim=True)
            self.sem_grid = mul_probs_grid / normalization_grid.repeat(1, self.object_labels, 1, 1)
            step_geo_grid[:,i,:,:,:] = self.sem_grid.clone()
        return step_geo_grid

   
    def update_uncertainty_map_avg(self, geo_grid):
        # Input geo_grid -- B x T x 1 x grid_dim x grid_dim
        # Update the uncertainty estimation at each location
        # Update only the locations where the geo_grid has uncertainty values
        for i in range(geo_grid.shape[1]):
            new_uncertainty_grid = geo_grid[:,i,:,:,:].clone()
            inds = torch.nonzero(new_uncertainty_grid > 1e-7, as_tuple=True)
            current_map = self.uncertainty_map.clone()
            current_map[inds[0],inds[1],inds[2],inds[3]] += new_uncertainty_grid[inds[0],inds[1],inds[2],inds[3]]
            current_map[inds[0],inds[1],inds[2],inds[3]] /= 2
            self.uncertainty_map = current_map


    def update_per_class_uncertainty_map_avg(self, geo_grid):
        # Input geo_grid -- B x T x C x grid_dim x grid_dim
        # Update the per class uncertainty estimation at each location
        # Update only the locations where the geo_grid has uncertainty values
        for i in range(geo_grid.shape[1]):
            new_uncertainty_grid = geo_grid[:,i,:,:,:].clone()
            inds = torch.nonzero(new_uncertainty_grid > 1e-7, as_tuple=True)
            current_map = self.per_class_uncertainty_map.clone()
            current_map[inds[0],inds[1],inds[2],inds[3]] += new_uncertainty_grid[inds[0],inds[1],inds[2],inds[3]]
            current_map[inds[0],inds[1],inds[2],inds[3]] /= 2
            self.per_class_uncertainty_map = current_map


    def update_proj_grid_bayes(self, geo_grid):
        # Input geo_grid -- B x T (or 1) x num_of_classes x grid_dim x grid_dim
        # Update the ground-projected grid at each location
        step_geo_grid = torch.zeros((geo_grid.shape[0], geo_grid.shape[1], self.spatial_labels, 
                                            self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32).to(geo_grid.device)      
        for i in range(geo_grid.shape[1]): # sequence
            new_proj_grid = geo_grid[:,i,:,:,:]
            mul_proj_grid = new_proj_grid * self.proj_grid
            normalization_grid = torch.sum(mul_proj_grid, dim=1, keepdim=True)
            self.proj_grid = mul_proj_grid / normalization_grid.repeat(1, geo_grid.shape[2], 1, 1)
            step_geo_grid[:,i,:,:,:] = self.proj_grid.clone()
        return step_geo_grid


    def register_uncertainty(self, uncertainty_crop, pose, abs_pose):
        # used in the active training
        # assumes batch_size=1
        B, T, _, cH, cW = uncertainty_crop.shape
        ego_uncertainty_map = torch.zeros((T,1,self.grid_dim[0],self.grid_dim[1]), dtype=torch.float32, device=self.device)
        ego_uncertainty_map[:,:, self.crop_start:self.crop_end, self.crop_start:self.crop_end] = uncertainty_crop.squeeze(0)
        geo_uncertainty_maps = self.spatialTransformer(grid=ego_uncertainty_map, pose=pose, abs_pose=abs_pose)
        self.update_uncertainty_map_avg(geo_grid=geo_uncertainty_maps.unsqueeze(0)) # updates sg.uncertainty_map


    def register_per_class_uncertainty(self, per_class_uncertainty_crop, pose, abs_pose):
        B, T, C, cH, cW = per_class_uncertainty_crop.shape
        ego_per_class_uncertainty_map = torch.zeros((T,C,self.grid_dim[0],self.grid_dim[1]), dtype=torch.float32, device=self.device)
        ego_per_class_uncertainty_map[:,:, self.crop_start:self.crop_end, self.crop_start:self.crop_end] = per_class_uncertainty_crop.squeeze(0)
        geo_per_class_uncertainty_maps = self.spatialTransformer(grid=ego_per_class_uncertainty_map, pose=pose, abs_pose=abs_pose)
        self.update_per_class_uncertainty_map_avg(geo_grid=geo_per_class_uncertainty_maps.unsqueeze(0)) # updates sg.per_class_uncertainty_map


    def register_sem_pred(self, prediction_crop, pose, abs_pose):
        B, T, C, cH, cW = prediction_crop.shape
        ego_pred_map = torch.ones((T,C,self.grid_dim[0],self.grid_dim[1]), dtype=torch.float32, device=self.device) * (1/C)
        ego_pred_map[:,:, self.crop_start:self.crop_end, self.crop_start:self.crop_end] = prediction_crop.squeeze(0)
        geo_pred_map = self.spatialTransformer(grid=ego_pred_map, pose=pose, abs_pose=abs_pose)
        self.update_sem_grid_bayes(geo_grid=geo_pred_map.unsqueeze(0)) # updates sg.sem_grid
