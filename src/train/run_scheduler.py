# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:45:39 2021

@author: Divy
"""

from threading import Thread

import schedule

from src.train.train_run import run


def job():
    '''
    Wrapper function for driver function
    '''
    run()
    return


# Runs training every day at 1AM
schedule.every().day.at("01:00").do(job)


class MyClass(Thread):
    '''
    Thread to run training every day at 1AM in the background
    '''
    def __init__(self):
        '''
        Initialize thread
        '''
        Thread.__init__(self)
        self.daemon = True
        self.start()

    def run(self):
        '''
        Infinite loop to check if training should be run
        '''
        while True:
            schedule.run_pending()


if __name__ == "__main__":
    '''
    Main function to start the training thread
    '''
    MyClass()
