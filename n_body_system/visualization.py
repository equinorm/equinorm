import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import json




def draw_result(epochs, losses, local_path, title):
    print(epochs)
    print(losses)
    plt.plot(epochs, losses, '-b', label='gnn')
    #plt.plot(lst_iter, loss_egnn, '-r', label='egnn')
    #plt.plot(lst_iter, loss_baseline, '--g', label='baseline')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(local_path+'/'+title+".png")  # should before show method
    # show
    plt.show()


if __name__ == "__main__":
    local_path = '/apdcephfs_cq10/share_2934111/ziqiaomeng/SE3/'
    outf = 'n_body_system/logs'
    exp_name = 'exp_1_egnn_vel'
    norm_type = 'scale_norm'
    with open(local_path + outf + "/" + exp_name + norm_type + "1/train_losses.json",
              "r") as file:
        data = json.load(file)
        epochs = []
        losses = []
        for epoch, loss in data.items():
            epochs.append(int(epoch))
            loss = float("{:.2f}".format(loss))
            losses.append(loss)
    draw_result(epochs, losses, local_path, title='Comparison')
    #draw_result(epochs, gnn, egnn, baseline, title="Comparison")