#!/usr/bin/python


import matplotlib.pyplot as plt


def double_plot(original, final, titles=('', '', ''), output=''):
    """Plot the original and processed image together

    Parameters
    ----------
    original: numpy.ndarray
        Original image array.
    final: numpy.ndarray
        Final image array.
    titles: tuple
        Titles in the form ('original', 'final', 'super title')
    output: string
        Name of the output image if specified.
    """
    ft_size = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(original.shape) == 3:
        ax1.imshow(original)
    else:
        ax1.imshow(original, cmap='gray')
    ax1.set_title(titles[0], fontsize=ft_size)

    if len(final.shape) == 3:
        ax2.imshow(final)
    else:
        ax2.imshow(final, cmap='gray')
    ax2.set_title(titles[1], fontsize=ft_size)

    plt.suptitle(titles[2], fontsize=ft_size)
    plt.tight_layout()
    if output:
        plt.savefig(output)
        print("Image is saved at {}".format(output))
        plt.close()
    else:
        plt.show()


