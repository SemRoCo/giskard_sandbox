#!/usr/bin/env python

import yaml
import rospy
import rospkg
from giskard_msgs.srv import *
from giskard_msgs.msg import *

dummy_mode = False
dummy_db = {}

def get_ros_package_path():
    r = rospkg.RosPack()
    return r.get_path('giskard_task_database')

def read_dummy_db():
    filename = get_ros_package_path() + "/config/dummy_db.yaml"
    with open(filename, 'r') as stream:
        try:
            global dummy_db
            dummy_db = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def get_object_by_role(objs, role):
    for obj in objs:
        if obj.role == role:
            return obj

def get_source_object(objs):
    return get_object_by_role(objs, giskard_msgs.msg.TaskObject.SOURCE_ROLE)

def get_target_object(objs):
    return get_object_by_role(objs, giskard_msgs.msg.TaskObject.TARGET_ROLE)

def query_dummy_db(req):
    resp = QueryMotionResponse()
    target_obj = get_target_object(req.task.objects)
    source_obj = get_source_object(req.task.objects)
    global dummy_db

    for phase in dummy_db[req.task.type][source_obj.name][target_obj.name]:
        mp = MotionPhase(name=phase['name'])
        for constraint in phase['constraints']:
            c = Constraint(name=constraint['name'], lower=constraint['lower'], upper=constraint['upper'])
            mp.constraints.append(c)
        resp.phases.append(mp)
    return resp

def query_models(req):
    # todo: implement me
    return QueryMotionResponse()

def handle_query_motion(req):
    if dummy_mode:
        return query_dummy_db(req)
    else:
        return query_models(req)

def task_db_server():
    rospy.init_node('task_database')
    s = rospy.Service('~query_motion', QueryMotion, handle_query_motion)
    global dummy_mode
    dummy_mode = rospy.get_param('~dummy_mode')
    if dummy_mode:
        read_dummy_db()
    rospy.loginfo("dummy_mode: %s", dummy_mode)
    rospy.spin()

if __name__ == '__main__':
    task_db_server()
