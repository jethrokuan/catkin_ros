#!/usr/bin/env python
import rospy
from biotac_sensors.msg import SignedBioTacHand
from std_msgs.msg import Float64, Bool, String
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as inputMsg



# reset and activate


class PID_HELPER():
    def __init__(self):
        self.GOAL = 80 # in terms of desired pressure 230, 80
        self.TOLERANCE = 10
        self.TOLERANCE_QTY = 10

        self.input_topic = rospy.get_param("~input", "Robotiq2FGripperRobotInput")
        self.output_topic = rospy.get_param("~output", "Robotiq2FGripperRobotOutput")

        self.state=0
        self.current_pos=0
        rospy.init_node('pid_helper')
        self.pub = rospy.Publisher('state', Float64, queue_size=100)
        self.pub_goal = rospy.Publisher('setpoint', Float64, queue_size=100)
        self.pub_plant = rospy.Publisher(self.output_topic, outputMsg.Robotiq2FGripper_robot_output, queue_size=100)
        self.pub_pid_start = rospy.Publisher('pid_enable', Bool, queue_size=100)
        rospy.Subscriber(self.input_topic, inputMsg.Robotiq2FGripper_robot_input, self.getStatus)

        # command to be sent
        self.command = outputMsg.Robotiq2FGripper_robot_output();
        self.command.rACT = 0 #  1: activate the gripper, 0: reset the gripper -> try to activate the gripper from outside
        self.command.rGTO = 0 # Go To action: 0 or 1, 1 is action is taken
        self.command.rATR = 0 # Automatic Realease -> no need for now
        self.command.rPR = 0 # Desired target
        self.command.rSP = 0 # Desired speed: keep 0
        self.command.rFR = 0 # Desired force: keep 0

        self.init_gripper()
        self.pub_pid_start.publish(Bool(data=0))
        # start with msg
        #rospy.Subscriber('talkPID', String, self.callbackPID)

    #def callbackPID(self, data):
    #    if data.data == 'start':
    #        self.pub_pid_start.publish(Bool(data=1))


    def getStatus(self, status):
        self.current_pos = status.gPO
        #if self.current_pos >= self.GOAL:
        #    self.current_pos = self.GOAL


    def init_gripper(self):
        self.command.rACT = 0
        self.pub_plant.publish(self.command)
        rospy.sleep(0.1)
        self.command.rACT = 1
        self.command.rGTO = 1

        self.pub_plant.publish(self.command)
        print('Activated')

        # wait until open
        rospy.sleep(2)

        # send goal stuff
        self.pub_goal.publish(Float64(data=self.GOAL))
        print('Goal set')


    def updateState(self,data):
        self.state = data.bt_data[0].pdc_data

        if (abs(self.state - self.GOAL) < self.TOLERANCE) and (self.TOLERANCE_QTY != 0):
            #self.state = self.GOAL
            #self.pub_pid_start.publish(Bool(data=0)) 
            self.TOLERANCE_QTY -= 1
        if self.TOLERANCE_QTY == 0:
            self.pub_pid_start.publish(Bool(data=0))

        self.pub.publish(self.state)

    def updatePlant(self,data):
        action = self.current_pos + data.data
        print('Input to the plant:', data.data)
        self.command.rPR = action
        print(action)
        self.pub_plant.publish(self.command)


    def listener(self): 
        rospy.Subscriber('biotac_pub_centered', SignedBioTacHand, self.updateState)
        rospy.Subscriber('control_effort', Float64, self.updatePlant)
        rospy.spin()

if __name__ == '__main__':
    my_helper = PID_HELPER()
    rospy.sleep(0.1)
    my_helper.listener()
