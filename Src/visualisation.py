from . import *

def flow_vis(a,b,X,xlimit,ylimit,title):
    fig, ax = plt.subplots(2,2)
    cmap = []

    # Overview
    cmap.append(ax[0,0].tricontour(a.squeeze(),b.squeeze(),X.squeeze(),levels=5))
    cmap.append(ax[0,1].tricontourf(a.squeeze(),b.squeeze(),X.squeeze(),levels=75))

    # Boundary layer initiation (zoomed in version)
    cmap.append(ax[1,0].tricontour(a.squeeze(),b.squeeze(),X.squeeze(),levels=5))
    ax[1,0].set_xlim(xlimit)
    ax[1,0].set_ylim(ylimit)

    cmap.append(ax[1,1].tricontourf(a.squeeze(),b.squeeze(),X.squeeze(),levels=75))
    ax[1,1].set_xlim(xlimit)
    ax[1,1].set_ylim(ylimit)

    # Figure format
    fig.suptitle(title,fontsize=14)
    fig.supxlabel("X-coordinate",fontsize=14)
    fig.supylabel("Z-coordinate",fontsize=14)
    for i in range(len(cmap)):
        fig.colorbar(cmap[i])
    fig.tight_layout()
    fig.savefig(f"Visualisation/{title}",transparent=True)


def compare(x_test,Y_test,y_pred):
    # Full contour plot
    # Y_test = scaler.inverse_transform(Y_test)

    fig, ax = plt.subplots(1,2,figsize=(8,3))
    cmap_U_pred_full = []

    cmap_U_pred_full.append(ax[0].tricontourf(x_test[:,0].detach().numpy(),x_test[:,1].detach().numpy(),Y_test[:,0].detach().numpy(),levels=1025))
    cmap_U_pred_full.append(ax[1].tricontourf(x_test[:,0].detach().numpy(),x_test[:,1].detach().numpy(),y_pred[:,0].detach().numpy(),levels=1025))
    for i in range(len(cmap_U_pred_full)):
        fig.colorbar(cmap_U_pred_full[i])
        
    fig.suptitle("Flat plate true and predicted flow field (U-Velocity)")
    fig.supxlabel("X-coordinate",fontsize=14)
    fig.supylabel("Y-coordinate",fontsize=14)
    plt.tight_layout()
    fig.savefig("Visualisation/Flat prediction.png",transparent=True)