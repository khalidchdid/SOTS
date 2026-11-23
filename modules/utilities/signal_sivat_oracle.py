import os
import h5py
import matplotlib.pyplot as plt
import numpy as np



class CSignalSivatOracle:
    """
    This is a class that manipulates a choosed signal :
        13/03/2025 : it display the signals sivat, oracle, and validities        
    """
    def __init__(self, path_oracle_sivat, str_signal_name = "n_SI43_D1_E24_A21"):
        self.str_signal_name = str_signal_name
        self.path_oracle_sivat = path_oracle_sivat
        self._get_data()
    def __str__(self):
        return "< " + self.str_signal_name + " >"
    

    def _get_data(self):
        # read oracle.h5 & sivat.h5 file ( found in rule folder of a compain folder)
        # that contain multiple signals
        
        oracle_path = os.path.join(self.path_oracle_sivat, 'oracle.h5')
        sivat_path  = os.path.join(self.path_oracle_sivat, 'sivat.h5')
        # read h5file
        oracle = h5py.File(oracle_path, mode = 'r')
        sivat  = h5py.File(sivat_path, mode = 'r')
        
        # get outputs   
        oracle_outpt_grp = oracle.get('outputs')
        sivat_outpt_grp = sivat.get('outputs')
        oracle_time_grp = oracle.get('time')
        sivat_time_grp = sivat.get('time')

        # get group
        oracle_signal = oracle_outpt_grp[self.str_signal_name]
        oracle_time = oracle_time_grp['n_times']
        sivat_signal = sivat_outpt_grp[self.str_signal_name]
        sivat_time = sivat_time_grp['n_times']

                
        # get values & validities
        self.list_oracle_sgl_values = oracle_signal[0,:]
        self.list_oracle_sgl_validities = [int(x) for x in oracle_signal[1,:]]
        self.list_oracle_time = np.array(oracle_time).tolist()

        self.list_sivat_sgl_values = sivat_signal[0,:]
        self.list_sivat_sgl_validities = [int(x) for x in sivat_signal[1,:]]
        self.list_sivat_time = np.array(sivat_time).tolist()

        self.sivat_shape = self.list_sivat_sgl_values.shape[0]
        self.oracle_shape = self.list_oracle_sgl_values.shape[0]

        
        
        #  The normalized time is the vecteur of time from 0 to 1 corresponding to the sampling
        self.list_time_normalized_oracle = np.arange(0.0, 1.0, 1/len(self.list_oracle_sgl_values))
        self.list_time_normalized_sivat = np.arange(0.0, 1.0, 1/len(self.list_sivat_sgl_values))
        
        
    # getters functions
    def get_sivat_values(self):
        """ list 1-D values """
        return self.list_sivat_sgl_values
    def get_oracle_values(self):
        """ list 1-D values """
        return self.list_oracle_sgl_values
    def get_sivat_validities(self):
        """ list 1-D validities of int 0 or 1 """
        return self.list_sivat_sgl_validities
    def get_oracle_validities(self):
        """ list 1-D validities of int 0 or 1 """
        return self.list_oracle_sgl_validities
    def get_sivat_shape(self):
        """ return : scalar int """
        return self.sivat_shape
    def get_oracle_shape(self):
        """ return : scalar int """
        return self.oracle_shape
    def get_oracle_time(self):
        """ return : scalar int """
        return self.list_oracle_time
    def get_sivat_time(self):
        """ return : scalar int """
        return self.list_sivat_time
    def get_oracle_time_normalized(self):
        """ return : scalar int """
        return self.list_time_normalized_oracle
    def get_sivat_time_normalized(self):
        """ return : scalar int """
        return self.list_time_normalized_sivat
    

    # plot
    
    def plot_signal(self, plot_dir, dpi = 1200, list_t_oracle = [], list_t_sivat = []):
        if list_t_oracle == []:
            list_t_oracle = self.list_time_normilized_oracle
        if list_t_sivat == []:
            list_t_sivat = self.list_time_normilized_sivat

            
        ouput_img_path = os.path.join(plot_dir, self.str_signal_name + ".png")

        fig, axs = plt.subplots(2,1, sharex=True)
        axs[0].plot(list_t_oracle, self.list_oracle_sgl_values, linewidth = 10, c = "r", zorder=0, alpha = 0.5)
        axs[0].plot(list_t_sivat, self.list_sivat_sgl_values, linewidth = 5, c = "b", zorder=1, alpha = 0.5)

        axs[0].scatter(list_t_oracle, self.list_oracle_sgl_values, marker = '.', s = 2, c = "r", label = "oracle", zorder = 2,edgecolor='none')
        axs[0].scatter(list_t_sivat, self.list_sivat_sgl_values, marker = '.', s = 1, c = "b", label = "sivat", zorder = 3, edgecolor='none')
        axs[0].grid(color='black', linestyle='-', axis = "x", linewidth=2, zorder = 4, alpha = 0.5)

        axs[0].legend(markerscale = 5)
        axs[1].plot(list_t_oracle, self.list_oracle_sgl_validities, c = "r", linewidth = 10.0)
        axs[1].plot(list_t_sivat, self.list_sivat_sgl_validities, c = "b", linewidth = 5.0)
        axs[1].set(ylim = (-0.1,1.1))
        axs[1].grid(color='black', linestyle='-', axis = "x", linewidth=2, zorder = 4, alpha = 0.5)

        # dpi is the resolution
        plt.savefig(ouput_img_path,dpi = dpi)
        plt.close()
        del fig, axs
        # plt.show()