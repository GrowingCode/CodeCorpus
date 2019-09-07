package com.zzk;

import java.applet.Applet;
import java.awt.Color;
import java.awt.Graphics;

public class ShowParamApplet extends Applet {
    String id;// 声明成员变量
    String name;// 声明成员变量
    String sex;// 声明成员变量
    public void init() {
        id = getParameter("id");// 获得参数id的值
        name = getParameter("name");// 获得参数name的值
        sex = getParameter("sex");// 获得参数sex的值
    }
    public void paint(Graphics g) {
        g.setColor(Color.blue);// 设置颜色
        g.drawString("学号：", 30, 10);// 绘制文本
        g.drawString(id, 70, 10);// 绘制获得的id值
        g.drawString("姓名：", 30, 30);// 绘制文本
        g.drawString(name, 70, 30);// 绘制获得的name值
        g.drawString("性别：", 30, 50);// 绘制文本
        g.drawString(sex, 70, 50);// 绘制获得的sex值
    }
}
