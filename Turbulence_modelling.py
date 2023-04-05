from Src import *
from Src import read_data
from Src import model
from Src import train
from Src import visualisation


def main():
    # Import and augment data
    da = read_data.data_augmentation()
    da.compile()

    # Physics informed neural network
    Model = model.PINN(da)
    Model.float().to(device)
    print(Model)

    # Model training
    training = train.train(da,Model)
    training.train()
    training.predict()
    training.extrapolate()

    # Visualisation
    visualisation.compare(training.da.x_test,training.da.Y_test,training.da.scaler.inverse_transform(training.y_pre))
    title = "Contour plot for predicted U (x-component velocity)"
    xlim = [-0.015,0.05]
    ylim = [0,0.0005]
    visualisation.flow_vis(training.da.X_test[:,0].detach(),training.da.X_test[:,1].detach(),training.y_pred[:,0].detach(),xlim,ylim,title)
    title = "Contour plot for predicted Pressure"
    xlim = [-0.03,0.05]
    ylim = [0,0.042]
    visualisation.flow_vis(training.da.X_test[:,0].detach(),training.da.X_test[:,1].detach(),training.y_pred[:,2].detach(),xlim,ylim,title)
    plt.show()

if __name__ == '__main__':
    main()