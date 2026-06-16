import os
import numpy as np
from scipy.io import loadmat
import scipy.spatial.distance


class DensePoseMethods:
    def __init__(self):
        # Load the UV data from the specified .mat file
        ALP_UV = loadmat(os.path.join('./data/UV_data', 'UV_Processed.mat'))
        
        # Extract and process the face indices, faces, and UV coordinates
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1  # MATLAB indices start at 1, so we subtract 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        
        # Info to compute symmetries
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
        
        # Load the UV symmetry transformations
        UV_symmetry_filename = os.path.join('./data/UV_data', 'UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        # Initialize symmetric labels and UV coordinates
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        
        # Iterate over possible labels
        for i in range(24):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                
                # Convert UV coordinates to integer indices
                U_loc = (U[jj] * 255).astype(np.int64)
                V_loc = (V[jj] * 255).astype(np.int64)
                
                # Apply symmetry transformations
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        
        # Flip the mask horizontally and apply symmetries
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        for i in range(14):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        
        # Compute the symmetric coordinates
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x
        
        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        # Compute vectors from P0 to P1, P2, and P
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        
        # Check if P is inside the triangle using cross products
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if np.dot(vCrossW, vCrossU) < 0:
            return False
        
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        if np.dot(uCrossW, uCrossV) < 0:
            return False
        
        # Compute barycentric coordinates
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        
        # Check if the point is within the triangle
        return (r <= 1) & (t <= 1) & (r + t <= 1)

    def barycentric_coordinates(self, P0, P1, P2, P):
        # Compute vectors from P0 to P1, P2, and P
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        
        # Compute cross products
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        
        # Compute barycentric coordinates
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        
        return 1 - (r + t), r, t

    def IUV2FBC(self, I_point, U_point, V_point):
        # Convert UV point to 3D point
        P = [U_point, V_point, 0]
        
        # Get faces corresponding to the given index
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        
        # Get the UV coordinates of the vertices of these faces
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0], np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1], np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2], np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()
        
        # Check if the UV point is within any of the faces using barycentric coordinates
        for i, (P0, P1, P2) in enumerate(zip(P_0, P_1, P_2)):
            if self.barycentric_coordinates_exists(P0, P1, P2, P):
                bc1, bc2, bc3 = self.barycentric_coordinates(P0, P1, P2, P)
                return FaceIndicesNow[0][i], bc1, bc2, bc3
        
        # If the UV point is not inside any faces, find the closest vertex
        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        
        if (minD1 < minD2) & (minD1 < minD3):
            return FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.
        elif (minD2 < minD1) & (minD2 < minD3):
            return FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.
        else:
            return FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):
        # Get vertex indices for the given face
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        
        # Compute the 3D point on the surface using barycentric coordinates
        p = Vertices[Vert_indices[0], :] * bc1 + Vertices[Vert_indices[1], :] * bc2 + Vertices[Vert_indices[2], :] * bc3
        
        return (p)

