from . import *
from . import device
from . import config

class TorchStandardScaler():
    def __init__(self):
        self.mean = 0
        self.sd = 0
        self.fitted = False
        
    def fit(self,input_tensor):
        self.mean = torch.mean(input_tensor,0)
        self.std = torch.std(input_tensor,0)
        
    def transform(self,input_tensor):
        return (input_tensor-self.mean)/self.std
    
    def inverse_transform(self,input_tensor):
        return input_tensor*self.std+self.mean
    
class data_augmentation:

    def __init__(self):
        self.filepath = "Data/flatplate_SA.csv"

    def read_data(self):
        df_raw = pd.read_csv(self.filepath)
        self.df = df_raw[df_raw.z<0.05]
        print(f"data shape : {self.df.shape}")
    
    def data_extract(self):
        ## Coordinates
        self.x = torch.tensor(np.array(self.df["x"])).view(-1,1)
        self.z = torch.tensor(np.array(self.df["z"])).view(-1,1)
        self.U = torch.tensor(np.array(self.df["U"])).view(-1,1)
        self.W = torch.tensor(np.array(self.df["W"])).view(-1,1)
        self.P = torch.tensor(np.array(self.df["Pressure"])).view(-1,1)
        self.Nu_tilde = torch.tensor(np.array(self.df["Nu_Tilde"])).view(-1,1)

        self.Rho = torch.tensor(np.array(self.df["Density"])).view(-1,1)

        # All data available
        self.x_test = torch.hstack([self.x,self.z]) # The coordinates
        self.y_test = torch.hstack([self.U,self.W,self.P,self.Nu_tilde]) # Flow parameters

        self.scaler = TorchStandardScaler()
        self.scaler.fit(self.y_test)
        self.y_test = self.scaler.transform(self.y_test)

        # Domain range
        self.x_lb,self.x_ub,self.z_lb,self.z_ub = torch.min(self.x_test[:,0]),torch.max(self.x_test[:,0]),torch.min(self.x_test[:,1]),torch.max(self.x_test[:,1])
        # Velocity range
        U_lb,U_ub,W_lb,W_ub = torch.min(self.y_test[:,0]),torch.max(self.y_test[:,0]),torch.min(self.y_test[:,1]),torch.max(self.y_test[:,1])
        P_lb,P_ub,Nu_lb,Nu_ub = torch.min(self.y_test[:,2]),torch.max(self.y_test[:,2]),torch.min(self.y_test[:,3]),torch.max(self.y_test[:,3])

        ## Note testing set is unscaled 
        # Boundary condition

        # At the wall
        self.wall_mask = (self.x_test[:,0]>=0).numpy() & (self.x_test[:,1]==self.z_lb).numpy() # x >= 0 and z = 0
        self.x_train_wall = self.x_test[self.wall_mask]
        self.y_train_wall = self.y_test[self.wall_mask]

        # Inlet
        self.inlet_mask = (self.x_test[:,0]==self.x_lb).numpy() # x = -0.333
        self.x_train_inlet = self.x_test[self.inlet_mask]
        self.y_train_inlet = self.y_test[self.inlet_mask]

        # Outlet
        self.outlet_mask = (self.x_test[:,0]==self.x_ub).numpy() | (self.x_test[:,1]==self.z_ub).numpy() # x = 2 and z = 1
        self.x_train_outlet = self.x_test[self.outlet_mask]
        self.y_train_outlet = self.y_test[self.outlet_mask]

        # Boundary condition data set
        self.x_train_BC = torch.vstack([self.x_train_wall,self.x_train_inlet,self.x_train_outlet])
        self.y_train_BC = torch.vstack([self.y_train_wall,self.y_train_inlet,self.y_train_outlet])
        # Shuffling
        self.idx_BC = np.random.choice(self.x_train_BC.shape[0],size=config.N_BC,replace=False)
        self.x_train_BC = self.x_train_BC[self.idx_BC,:]
        self.y_train_BC = self.y_train_BC[self.idx_BC,:]

        # Collocation points (Full RANS equation)
        self.lower_bound = torch.hstack([self.x_lb,self.z_lb])
        self.upper_bound = torch.hstack([self.x_ub,self.z_ub])
        self.x_train_RANS = self.lower_bound + (self.upper_bound-self.lower_bound)*lhs(2,samples=config.N_RANS)
        self.x_train_RANS = torch.vstack([self.x_train_RANS,self.x_train_BC])

        # Sample points from dataset
        self.idx_sample = np.random.choice(self.x_test.shape[0],size=config.N_sample,replace=False)
        self.x_train_sample = self.x_test[self.idx_sample,:]
        self.y_train_sample = self.y_test[self.idx_sample,:]

    def load_to_device(self):
        # Boundary condition
        self.X_train_BC = (self.x_train_BC).float().to(device)
        self.Y_train_BC = (self.y_train_BC).float().to(device)

        # Internal sampling
        self.X_train_sample = (self.x_train_sample).float().to(device)
        self.Y_train_sample = (self.y_train_sample).float().to(device)

        # Governing equation (RANS equation)
        self.X_train_RANS = (self.x_train_RANS).float().to(device)
        self.RANS_hat = torch.zeros(self.X_train_RANS.shape[0],1).float().to(device) # Zero

        # Testing
        self.X_test = (self.x_test).to(device)
        self.Y_test = (self.y_test).float().to(device)

    def compile(self):
        self.read_data()
        self.data_extract()
        self.load_to_device()       