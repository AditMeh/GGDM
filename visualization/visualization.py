import imageio
import matplotlib.pyplot as plt
import matplotlib

def plot_two_rows(images_lst_1, images_lst_2, time_lst, save_file):
    num_per_row = len(time_lst)

    f, axarr = plt.subplots(2, num_per_row)
    f.set_size_inches(26, 4.76)    
    f.subplots_adjust(hspace=0, wspace=0)
    

    for i in range(num_per_row):
        title = f'T={time_lst[i]}'
        
        # first row
        (axarr[0][i]).imshow(images_lst_1[i])
        (axarr[0][i]).title.set_text(title)
        (axarr[0][i]).set_xticks([])
        (axarr[0][i]).set_yticks([])
        (axarr[0][i]).set_xticklabels([])
        (axarr[0][i]).set_yticklabels([])
        axarr[0][i].spines['top'].set_visible(False)
        axarr[0][i].spines['right'].set_visible(False)
        axarr[0][i].spines['bottom'].set_visible(False)
        axarr[0][i].spines['left'].set_visible(False)

        for item in ([axarr[0][i].title, axarr[0][i].xaxis.label, axarr[0][i].yaxis.label] +
                    axarr[0][i].get_xticklabels() + axarr[0][i].get_yticklabels()):
            item.set_fontsize(20)



        # second row
        (axarr[1][i]).imshow(images_lst_2[i])
        (axarr[1][i]).set_xticks([])
        (axarr[1][i]).set_yticks([])
        (axarr[1][i]).set_xticklabels([])
        (axarr[1][i]).set_yticklabels([])
        axarr[1][i].spines['top'].set_visible(False)
        axarr[1][i].spines['right'].set_visible(False)
        axarr[1][i].spines['bottom'].set_visible(False)
        axarr[1][i].spines['left'].set_visible(False)

        for item in ([axarr[1][i].title, axarr[1][i].xaxis.label, axarr[1][i].yaxis.label] +
                    axarr[1][i].get_xticklabels() + axarr[1][i].get_yticklabels()):
            item.set_fontsize(20)


    (axarr[0][0]).set_ylabel(r'$x_t$       ', rotation="horizontal")
    (axarr[1][0]).set_ylabel(r'$\hat x_0 (x_t)$            ', rotation="horizontal")

    f.savefig(save_file)


def plot_CelebA():
    time_lst = [1000 - i * 100 for i in range(10)]
    time_lst.append(1)

    images_lst_1 = []
    images_lst_2 = []

    for time in time_lst:
        images_lst_1.append(imageio.imread(f'/voyager/projects/aditya/DDPM-sandbox/celba/{time}.png'))
        images_lst_2.append(imageio.imread(f'/voyager/projects/aditya/DDPM-sandbox/celba_tweedle/tweedle_{time}.png'))

    plot_two_rows(images_lst_1, images_lst_2, time_lst, 'CelebA.pdf')


def plot_MNIST_classifier():
    time_lst = [1000 - i * 100 for i in range(10)]
    time_lst.append(1)

    images_lst_1 = []
    images_lst_2 = []

    for time in time_lst:
        images_lst_1.append(imageio.imread(f'/voyager/projects/aditya/DDPM-sandbox/celba/{time}.png'))
        images_lst_2.append(imageio.imread(f'/voyager/projects/aditya/DDPM-sandbox/celba_tweedle/tweedle_{time}.png'))

    plot_two_rows(images_lst_1, images_lst_2, time_lst, 'MNIST_classifier.png')


plot_MNIST_classifier()