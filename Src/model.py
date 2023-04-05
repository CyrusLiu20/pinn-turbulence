from . import *
from . import config

# Spalart-Allmaras turbullence model
sigma = 2/3
C_b1 = 0.1355
C_b2 = 0.622
k = 0.41
C_w1 = C_b1/k**2  + (1+C_b2)/sigma**2
C_w2 = 0.3
C_w3 = 2
C_nu1 = 7.1

# Wall function
chi = lambda nu_tilde : nu_tilde/config.nu_laminar 
f_v1 = lambda nu_tilde : chi(nu_tilde)**3 / (chi(nu_tilde)**3 + C_nu1**3)
nu_tilde_to_eddy = lambda nu_tilde : nu_tilde*f_v1(nu_tilde)

class PINN(nn.Module):
    def __init__(self,da):
        super().__init__() # initialising the parent class
        self.activation_relu = nn.ReLU()
        self.activation = nn.Tanh()
        self.layers = config.layers
        self.da = da
        self.loss_function = nn.MSELoss(reduction="mean")
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        self.iter = 0
        # Weight and bias initalisation
        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data,gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data) # All bias set to zero
            
    # Forward passes
    def forward(self,x):
        if torch.is_tensor(x) != True:
            a = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        
        # learned features
        z = self.linears[-1](a) # ReLu
        a = self.activation_relu(z)
        return a
    
    # Boundary condition
    def loss_inlet(self,x_inlet,y_inlet):
        g = x_inlet.clone()
        g.requires_grad = True
        f_raw = self.forward(g)

        f = self.da.scaler.inverse_transform(f_raw)
        y_inlet = self.da.scaler.inverse_transform(y_inlet)

        loss_inlet_velocity_u = self.loss_function(f[:,0],y_inlet[:,0])
        loss_inlet_velocity_v = self.loss_function(f[:,1],y_inlet[:,1])
        
        return loss_inlet_velocity_u + loss_inlet_velocity_v


    def loss_outlet(self,x_outlet,y_outlet):
        f_raw = self.forward(x_outlet)
        f = self.da.scaler.inverse_transform(f_raw)
        y_outlet = self.da.scaler.inverse_transform(y_outlet)
        
        loss_outlet_velocity_u = self.loss_function(f[:,0],y_outlet[:,0])
        loss_outlet_velocity_v = self.loss_function(f[:,1],y_outlet[:,1])
        
        return loss_outlet_velocity_u + loss_outlet_velocity_v
    
    def loss_wall(self,x_wall,y_wall):
        g = x_wall.clone()
        g.requires_grad = True
        f_raw = self.forward(g)
        f = self.da.scaler.inverse_transform(f_raw).float().to(device)
        y_wall = self.da.scaler.inverse_transform(y_wall).float().to(device)
        zeros = (torch.ones([f.shape[0]])*0).float().to(device)
        
        loss_no_slip_x = self.loss_function(f[:,0],zeros)
        loss_no_slip_z = self.loss_function(f[:,1],zeros)
        loss_wall_viscosity_tilde = self.loss_function(f[:,3],zeros)
       
        return loss_no_slip_x + loss_no_slip_z + loss_wall_viscosity_tilde# + loss_normal_pressure_gradient
    
    # Governing equation
    def loss_turbulent(self, x_turbulent):
        g = x_turbulent.clone()
        g.requires_grad = True
        f_raw = self.forward(g)
        f = (self.da.scaler.inverse_transform(f_raw))
        U_xz = autograd.grad(f[:,0][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        U_xx_zz = autograd.grad(U_xz,g,torch.ones(g.shape).to(device),create_graph=True)[0]
        U_x = U_xz[:,[0]]
        U_z = U_xz[:,[1]]
        U_xx = U_xx_zz[:,[0]]
        U_zz = U_xx_zz[:,[1]]
        W_xz = autograd.grad(f[:,1][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        W_xx_zz = autograd.grad(W_xz,g,torch.ones(g.shape).to(device),create_graph=True)[0]
        W_x = W_xz[:,[0]]
        W_z = W_xz[:,[1]]      
        W_xx = W_xx_zz[:,[0]]
        W_zz = W_xx_zz[:,[1]]
        P_xz = autograd.grad(f[:,2][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        P_x = P_xz[:,[0]]
        P_z = P_xz[:,[1]]        
        
        nu_eddy = nu_tilde_to_eddy(f[:,[3]])
        nu = nu_eddy + config.nu_laminar

        Nu_tilde_xz = autograd.grad(f[:,3][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        Nu_eddy_xz = autograd.grad(nu_eddy,g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        Nu_tilde_x = Nu_tilde_xz[:,[0]]
        Nu_tilde_z = Nu_tilde_xz[:,[1]]
        Nu_eddy_x = Nu_eddy_xz[:,[0]]
        Nu_eddy_z = Nu_eddy_xz[:,[1]]

        # Spalart-Allmarus
        nu_tilde = f[:,[3]].float()
        f_v2 = 1 - chi(nu_tilde)/(1+chi(nu_tilde)*f_v1(nu_tilde))
        d = g[:,[1]]
        omega = 0.5 * (U_z - W_x)     
        S = torch.sqrt(2*omega*omega)
        
        S_tilde = S + nu_tilde/(k**2*d**2)*f_v2
        # if S_tilde < 1e-6:
        #     print(S,nu_tilde,f_v2)
        S_tilde = torch.clip(S_tilde,max=1e-6)
        r = torch.clip(nu_tilde/(S_tilde*k**2*d**2),min=10)
        # print(r)
        # S_tilde = S + nu_tilde/(k**2*d**2)*f_v2
        # r = nu_tilde/(S_tilde*k**2*d**2)
        g_SA = r + C_w2*(r**6-r)
        f_w = g_SA * ((1+C_w3**6)/(g_SA**6+C_w3**6))**(1/6)
        
        #################### Spallart-Allmarus model #######################
        RANS_SA = (f[:,[0]].float()*Nu_tilde_x + f[:,[1]].float()*Nu_tilde_z 
                   - C_b1*S_tilde*nu_tilde
                   + C_w1*f_w*(nu_tilde/d)**2
                   - 1/sigma*((Nu_eddy_x+Nu_tilde_x)*Nu_tilde_x + (Nu_eddy_z+Nu_tilde_z)*Nu_tilde_z)
                   - C_b2*(Nu_tilde_x*Nu_tilde_x+Nu_tilde_z*Nu_tilde_z))
        #################### Spallart-Allmarus model #######################
        
        zeros = torch.zeros(RANS_SA.shape[0],1).to(device)
        
        loss_SA = self.loss_function(RANS_SA,zeros) 
    
        return loss_SA
        
    def loss_RANS(self, x_RANS):
        g = x_RANS.clone()
        g.requires_grad = True
        f_raw = self.forward(g)
        # f = f_raw
        
        f = (self.da.scaler.inverse_transform(f_raw))
        U_xz = autograd.grad(f[:,0][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        U_xx_zz = autograd.grad(U_xz,g,torch.ones(g.shape).to(device),create_graph=True)[0]
        U_x = U_xz[:,[0]]
        U_z = U_xz[:,[1]]
        U_xx = U_xx_zz[:,[0]]
        U_zz = U_xx_zz[:,[1]]
        W_xz = autograd.grad(f[:,1][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        W_xx_zz = autograd.grad(W_xz,g,torch.ones(g.shape).to(device),create_graph=True)[0]
        W_x = W_xz[:,[0]]
        W_z = W_xz[:,[1]]      
        W_xx = W_xx_zz[:,[0]]
        W_zz = W_xx_zz[:,[1]]
        P_xz = autograd.grad(f[:,2][:,None],g,torch.ones([g.shape[0], 1]).to(device),retain_graph=True,create_graph=True)[0]
        P_x = P_xz[:,[0]]
        P_z = P_xz[:,[1]]
   
        nu_eddy = nu_tilde_to_eddy(f[:,[3]])
        nu = nu_eddy + config.nu_laminar
        
        #################### RANS #######################
        RANS_continuity = U_x + W_z
        
        RANS_momentum_x = ((f[:,[0]].float()*U_x + f[:,[1]].float()*U_z 
                           + 1/(config.density)*P_x - nu.float()*(U_xx+U_zz)))
        
        RANS_momentum_z = ((f[:,[0]].float()*W_x + f[:,[1]].float()*W_z 
                            + 1/(config.density)*P_z - nu.float()*(W_xx+W_zz)))
        #################### RANS #######################
        
        lambda_continuity, lambda_x, lambda_z = 1e3, 1, 1
        loss_continuity = self.loss_function(RANS_continuity,self.da.RANS_hat)
        loss_momentum_x = self.loss_function(RANS_momentum_x,self.da.RANS_hat)
        loss_momentum_z = self.loss_function(RANS_momentum_z,self.da.RANS_hat)
        
        return loss_continuity*lambda_continuity + loss_momentum_x*lambda_x + loss_momentum_z*lambda_z
    
    # Sample points
    def loss_sample(self,x_train_sample,y_train_sample):
        f_raw = self.forward(x_train_sample)      
        f = self.da.scaler.inverse_transform(f_raw)
        
        y_train_sample = self.da.scaler.inverse_transform(y_train_sample)     
        loss_sampling = self.loss_function(f,y_train_sample) 
        return loss_sampling
    
    # Boundary condition (wrapper function)
    def loss_BC(self,x_PINN_wall,y_PINN_wall,x_PINN_inlet,y_PINN_inlet,x_PINN_outlet,y_PINN_outlet):
        return self.loss_wall(x_PINN_wall,y_PINN_wall) + self.loss_inlet(x_PINN_inlet,y_PINN_inlet) + self.loss_outlet(x_PINN_outlet,y_PINN_outlet)
    
    def loss(self,mode,x_train_BC,y_train_BC,x_train_RANS,x_train_sample,y_train_sample):
        # At wall
        wall_mask = (x_train_BC[:,0]>=0).numpy() & (x_train_BC[:,1]==self.da.z_lb).numpy() # x >= 0 and z = 0
        x_PINN_wall = x_train_BC[wall_mask]
        y_PINN_wall = y_train_BC[wall_mask]
        # print(self.scaler.inverse_transform(y_PINN_wall))
        
        # At inlet
        inlet_mask = (x_train_BC[:,0]==self.da.x_lb).numpy() # x = -0.333
        x_PINN_inlet = x_train_BC[inlet_mask]
        y_PINN_inlet = y_train_BC[inlet_mask]
        
        # At outlet
        outlet_mask = (x_train_BC[:,0]==self.da.x_ub).numpy() | (x_train_BC[:,1]==self.da.z_ub).numpy() # x = 2 and z = 1
        x_PINN_outlet = x_train_BC[outlet_mask]
        y_PINN_outlet = y_train_BC[outlet_mask]
        
        # Turbulent
        turbulent_mask = (x_train_RANS[:,0]>=0).numpy() & (x_train_RANS[:,1]>0.01).numpy() # So not to create nan from 
        x_PINN_turbulent = x_train_RANS[turbulent_mask]
        
        
        lambda1, lambda2, lambda3, lambda4 = 1e-4, 1e6, 1e0, 1e-1

        if mode == 0:
            return (#self.loss_RANS(x_train_RANS)*lambda1
                    # + self.loss_turbulent(x_PINN_turbulent)*lambda2
                    + self.loss_BC(x_PINN_wall,y_PINN_wall,x_PINN_inlet,y_PINN_inlet,x_PINN_outlet,y_PINN_outlet)*lambda3
                    + self.loss_sample(x_train_sample,y_train_sample)*lambda4)
                   # )
        elif mode == 1:
            return (#self.loss_RANS(x_train_RANS)*lambda1
                    # + self.loss_turbulent(x_PINN_turbulent)*lambda2
                    # + self.loss_BC(x_PINN_wall,y_PINN_wall,x_PINN_inlet,y_PINN_inlet,x_PINN_outlet,y_PINN_outlet)*lambda3
                    + self.loss_sample(x_train_sample,y_train_sample)*lambda4)
                   # )
        elif mode == 2:
            return (self.loss_RANS(x_train_RANS)*lambda1
                    # + self.loss_turbulent(x_PINN_turbulent)*lambda2
                    + self.loss_BC(x_PINN_wall,y_PINN_wall,x_PINN_inlet,y_PINN_inlet,x_PINN_outlet,y_PINN_outlet)*lambda3
                    + self.loss_sample(x_train_sample,y_train_sample)*lambda4)
                   # )
        elif mode == 3:
            return (self.loss_RANS(x_train_RANS)*lambda1
                    + self.loss_turbulent(x_PINN_turbulent)*lambda2
                    + self.loss_BC(x_PINN_wall,y_PINN_wall,x_PINN_inlet,y_PINN_inlet,x_PINN_outlet,y_PINN_outlet)*lambda3
                    + self.loss_sample(x_train_sample,y_train_sample)*lambda4)
                   # )
        elif mode == 4:
            return (#self.loss_RANS(x_train_RANS)*lambda1
                    + self.loss_turbulent(x_PINN_turbulent)*lambda2
                    # + self.loss_BC(x_PINN_wall,y_PINN_wall,x_PINN_inlet,y_PINN_inlet,x_PINN_outlet,y_PINN_outlet)*lambda3
                    # + self.loss_sample(x_train_sample,y_train_sample)*lambda4)
                   )
        else:
            return 0
        
        def checkpoint(self,Model,scheduler,i,loss,test_loss,lambda1=1e-4,lambda2=1e10,lambda3=1e-1,lambda4=1e-1):
            torch.save(Model.state_dict(), f"Models/Test model/epoch_{i}.pt")
            
            print(i,
                " lr : ",np.round(scheduler.get_last_lr()[0],6),
                "Training loss :",np.round(loss.detach().cpu().numpy(),3),
                "Testing loss :",np.round(test_loss.detach().cpu().numpy(),2),
                "RANS loss :",np.round(Model.loss_RANS(self.da.X_train_RANS).detach().cpu().numpy()*lambda1,4),
                # "Turb loss :",np.round(Model.loss_turbulent(X_train_RANS[(x_train_RANS[:,0]>=0).numpy() & (x_train_RANS[:,1]>0.05).numpy()]).detach().cpu().numpy()*lambda2,4),
                "BC loss :",np.round(Model.loss_function(Model(self.da.X_train_BC)[:,0:2],self.da.Y_train_BC[:,0:2]).detach().cpu().numpy()*lambda3,4),
                "Sample loss :",np.round(Model.loss_sample(self.da.X_train_sample,self.da.self.da.Y_train_sample).detach().cpu().numpy()*lambda4,4))