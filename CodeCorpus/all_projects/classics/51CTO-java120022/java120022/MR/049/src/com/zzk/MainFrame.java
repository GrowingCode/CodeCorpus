package com.zzk;

import java.awt.BorderLayout;
import java.awt.EventQueue;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;

public class MainFrame extends JFrame {
    private static final long serialVersionUID = 6760471507923160452L;
    private JTextField codeText;
    private JPasswordField pwdText;
    private JTextField nameText;
    DisturbCodePanel imageCode = null;
    
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    MainFrame frame = new MainFrame();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    public MainFrame() {
        super();
        setResizable(false);
        setTitle("带干扰线的验证码");
        setBounds(100, 100, 426, 210);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        imageCode = new DisturbCodePanel();// 创建类的实例
        imageCode.setBounds(170, 85, 106, 35);// 设置位置
        getContentPane().add(imageCode); // 添加验证码
        
        final JPanel panel = new JPanel();
        panel.setLayout(null);
        getContentPane().add(panel, BorderLayout.CENTER);
        
        final JButton button = new JButton();
        button.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                if (imageCode != null) {
                    imageCode.draw(); // 调用方法生成验证码
                }
            }
        });
        button.setText("换一张");
        button.setBounds(301, 90, 94, 28);
        panel.add(button);
        
        final JLabel label = new JLabel();
        label.setText("用户名：");
        label.setBounds(29, 25, 66, 18);
        panel.add(label);
        
        final JLabel label_1 = new JLabel();
        label_1.setText("密   码：");
        label_1.setBounds(29, 59, 66, 18);
        panel.add(label_1);
        
        nameText = new JTextField();
        nameText.setBounds(85, 23, 310, 22);
        panel.add(nameText);
        
        pwdText = new JPasswordField();
        pwdText.setBounds(85, 57, 310, 22);
        panel.add(pwdText);
        
        final JLabel label_1_1 = new JLabel();
        label_1_1.setText("验证码：");
        label_1_1.setBounds(29, 95, 66, 18);
        panel.add(label_1_1);
        
        codeText = new JTextField();
        codeText.setBounds(85, 93, 77, 22);
        panel.add(codeText);
        
        final JButton button_1 = new JButton();
        button_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                String username = nameText.getText();// 从文本框中获取用户名
                String password = new String(pwdText.getPassword());// 从密码框中获取密码
                String code = codeText.getText();// 获得输入的验证码
                String info = "";// 用户登录信息
                // 判断用户名是否为null或空的字符串
                if (username == null || username.isEmpty()) {
                    info = "用户名为空！";
                }
                // 判断密码是否为null或空的字符串
                else if (password == null || password.isEmpty()) {
                    info = "密码为空！";
                }
                // 判断验证码是否为null或空的字符串
                else if (code == null || code.isEmpty()) {
                    info = "验证码为空！";
                }
                // 判断 验证码是否正确
                else if (!code.equals(imageCode.getNum())) {
                    info = "验证码错误！";
                }
                // 如果用户名与密码均为"mrsoft"，则登录成功
                else if (username.equals("mrsoft") && password.equals("mrsoft")) {
                    info = "恭喜，登录成功";
                } else {
                    info = "用户名或密码错误！";
                }
                JOptionPane.showMessageDialog(null, info);// 通过对话框弹出用户登录信息
            }
        });
        button_1.setText("登  录");
        button_1.setBounds(42, 134, 106, 28);
        panel.add(button_1);
        
        final JButton button_1_1 = new JButton();
        button_1_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                nameText.setText("");// 清除用户名文本框内容
                pwdText.setText("");// 清除密码文本框内容
                codeText.setText("");// 清除验证码文本框内容
            }
        });
        button_1_1.setText("重  置");
        button_1_1.setBounds(191, 134, 106, 28);
        panel.add(button_1_1);
    }
    
}