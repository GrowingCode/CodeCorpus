package com.zzk;

import javax.swing.JApplet;

public class UseArchivePropertyApplet extends JApplet {
    public void init() {
        // 使用与html文件关联的NewPanelApp.jar文件中的NewPanel类创建对象
        npanel.NewPanel npanel = new npanel.NewPanel();
        setLayout(null);// 设置容器为绝对布局
        npanel.setBounds(10, 10, 254, 167);// 设置NewPanel对象的显示位置和大小
        add(npanel);// 将NewPanel对象添加到容器中
    }
}
