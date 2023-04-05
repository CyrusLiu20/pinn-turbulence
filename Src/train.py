from . import *
from . import config

class train:

    def __init__(self,da,Model):
        self.da = da
        self.Model = Model
        
        self.history = torch.zeros(config.iterations,2)
        self.optimizer = torch.optim.Adam(Model.parameters(),lr=config.lr,amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,1000,0.9)
        self.history_csv = []

    def checkpoint(self,Model,scheduler,i,loss,test_loss,lambda1=1e-4,lambda2=1e10,lambda3=1e-1,lambda4=1e-1):
        torch.save(Model.state_dict(), f"Iterations/Test model/epoch_{i}.pt")
        
        # print(i,
        #     " lr : ",np.round(scheduler.get_last_lr()[0],6),
        #     "Training loss :",np.round(loss.detach().cpu().numpy(),3),
        #     "Testing loss :",np.round(test_loss.detach().cpu().numpy(),2),
        #     "RANS loss :",np.round(Model.loss_RANS(self.da.X_train_RANS).detach().cpu().numpy()*lambda1,4),
        #     # "Turb loss :",np.round(Model.loss_turbulent(X_train_RANS[(x_train_RANS[:,0]>=0).numpy() & (x_train_RANS[:,1]>0.05).numpy()]).detach().cpu().numpy()*lambda2,4),
        #     "BC loss :",np.round(Model.loss_function(Model(self.da.X_train_BC)[:,0:2],self.da.Y_train_BC[:,0:2]).detach().cpu().numpy()*lambda3,4),
        #     "Sample loss :",np.round(Model.loss_sample(self.da.X_train_sample,self.da.Y_train_sample).detach().cpu().numpy()*lambda4,4))
        
        epoch = [i,scheduler.get_last_lr()[0],loss.detach().cpu().numpy(),test_loss.detach().cpu().numpy(),
                 Model.loss_RANS(self.da.X_train_RANS).detach().cpu().numpy()*lambda1,
                 Model.loss_function(Model(self.da.X_train_BC)[:,0:2],self.da.Y_train_BC[:,0:2]).detach().cpu().numpy()*lambda3,
                 Model.loss_sample(self.da.X_train_sample,self.da.Y_train_sample).detach().cpu().numpy()*lambda4]
        self.history_csv.append(epoch)

    def train(self):

        for i in range(config.iterations):
            self.loss = self.Model.loss(2,self.da.X_train_BC,self.da.Y_train_BC,self.da.X_train_RANS,self.da.X_train_sample,self.da.Y_train_sample)
            self.history[i,:] = torch.tensor([i,self.loss])
            
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if i % (config.steps-1)==0:
                with torch.no_grad():
                    test_loss = self.Model.loss_function(self.da.scaler.inverse_transform(self.Model(self.da.X_test)),self.da.scaler.inverse_transform(self.da.Y_test))      
                self.checkpoint(self.Model,self.scheduler,i,self.loss,test_loss,lambda1=1e-4,lambda2=1e10,lambda3=1e-1,lambda4=1e-1)

        self.history_csv = np.array(self.history_csv).T
        self.save_history()
    
    def save_history(self):
        np.savetxt(
            'history.csv',
            np.rec.fromarrays(self.history_csv),
            # fmt=['%d', '%d','%d','%d','%d','%d','%d'],
            delimiter=',',
            header='Epoch,Learning rate,Training loss,Testing loss,RANS loss,BC loss,Sample loss',
            comments='',
        )

    def predict(self):
        self.y_pred = self.Model(self.da.X_test)
        self.y_pred = self.da.scaler.inverse_transform(self.y_pred)

    def extrapolate(self):

        ###########Preparing unseen data############
        x1_future = torch.linspace(self.da.x_lb,3,400)
        x2_future = torch.linspace(self.da.z_lb,self.da.z_ub,400)

        X1_future,X2_future = torch.meshgrid(x1_future,x2_future,indexing='ij')
        X_future = torch.hstack([X1_future.reshape(-1,1),X2_future.reshape(-1,1)])       

        ###########Predicting unseen data############
        self.y_pred_future = self.Model(X_future)
        self.y_pred_future = self.da.scaler.inverse_transform(self.y_pred_future)