package com.zzk;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import javax.swing.JButton;
import javax.swing.Icon;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;

public class UpdateAccessPictureFrame extends JFrame {
    private JTextField tf_id;
    private static final long serialVersionUID = 8339759858493972497L;
    private JTextField tf_name;
    private String picturePath = null;
    
    /**
     * Launch the application
     * 
     * @param args
     */
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    UpdateAccessPictureFrame frame = new UpdateAccessPictureFrame();
                    frame.setVisible(true);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }
    
    /**
     * Create the frame
     */
    public UpdateAccessPictureFrame() {
        super();
        setTitle("修改Access数据库中存储的图片");
        setBounds(100, 100, 369, 232);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        final JPanel panel = new JPanel();
        panel.setLayout(null);
        getContentPane().add(panel, BorderLayout.CENTER);
        
        final JLabel label_1 = new JLabel();
        label_1.setText("姓    名：");
        label_1.setBounds(26, 43, 66, 18);
        panel.add(label_1);
        
        final JLabel label_2 = new JLabel();
        label_2.setText("图    片：");
        label_2.setBounds(26, 114, 66, 18);
        panel.add(label_2);
        
        tf_name = new JTextField();
        tf_name.setBounds(83, 41, 245, 22);
        panel.add(tf_name);
        
        final JLabel lb_picture = new JLabel();
        lb_picture.setOpaque(true);
        lb_picture.setBackground(Color.PINK);
        lb_picture.setBounds(83, 67, 139, 117);
        panel.add(lb_picture);
        
        final JButton button_2 = new JButton();
        button_2.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser(); // 创建文件对话框
                // 创建文件过滤
                FileFilter filter = new FileNameExtensionFilter(
                        "图像文件(*.gif;*.jpg;*.jpeg;*.png)", "gif", "jpg", "jpeg",
                        "png");
                fileChooser.setFileFilter(filter); // 为文件对话框设置文件过滤器
                int returnValue = fileChooser.showOpenDialog(null);// 打开文件选择对话框
                if (returnValue == JFileChooser.APPROVE_OPTION) { // 判断是否选择了文件
                    File file = fileChooser.getSelectedFile(); // 获得文件对象
                    if (file.length() / 1024.0 > 50.0) {
                        JOptionPane.showMessageDialog(null, "请选择小于等于50KB的图片文件。");
                        return;
                    }
                    picturePath = file.getAbsolutePath();
                    Icon icon = new ImageIcon(picturePath);
                    Dimension size = lb_picture.getSize();
                    lb_picture.setIcon(icon);
                    lb_picture.setSize(size);
                }
            }
        });
        button_2.setText("选择图片");
        button_2.setBounds(231, 69, 97, 28);
        panel.add(button_2);
        
        final JButton button = new JButton();
        button.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                String name = tf_name.getText().trim();
                if (picturePath == null || picturePath.equals("")) {
                    JOptionPane.showMessageDialog(null, "请选择图片文件。");
                    return;
                }
                File file = new File(picturePath);
                try {
                    FileInputStream in = new FileInputStream(file);// 创建图片的输入流对象
                    Connection conn = DAO.getConn();// 获得数据库连接对象
                    String sql = "update tb_picture set name = ?,picture = ? where id = ?";// 定义修改数据记录的SQL语句
                    PreparedStatement ps = conn.prepareStatement(sql);// 创建PreparedStatement对象，并传递SQL语句
                    ps.setString(1, name); // 为第1个参数赋值
                    ps.setBinaryStream(2, in, (int) file.length());// 为第2个参数赋值
                    ps.setInt(3, Integer.parseInt(tf_id.getText()));// 为第3个参数赋值
                    int flag = ps.executeUpdate(); // 执行SQL语句，获得更新记录数
                    if (flag > 0) {
                        JOptionPane.showMessageDialog(null, "修改成功！");
                    } else {
                        JOptionPane.showMessageDialog(null, "修改失败！");
                    }
                    ps.close();// 关闭PreparedStatement对象
                    conn.close(); // 关闭连接
                    if (in != null) {
                        in.close();// 关闭IO流
                    }
                } catch (FileNotFoundException e1) {
                    e1.printStackTrace();
                } catch (SQLException e2) {
                    e2.printStackTrace();
                } catch (IOException e3) {
                    e3.printStackTrace();
                }
            }
        });
        button.setBounds(231, 109, 97, 28);
        panel.add(button);
        button.setText("修    改");
        
        final JButton button_1 = new JButton();
        button_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                System.exit(0);
            }
        });
        button_1.setBounds(231, 151, 97, 28);
        panel.add(button_1);
        button_1.setText("退    出");

        tf_id = new JTextField();
        tf_id.setText("1");
        tf_id.setBounds(83, 10, 122, 22);
        panel.add(tf_id);

        final JButton button_3 = new JButton();
        button_3.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                int id = Integer.parseInt(tf_id.getText().trim());
                Connection conn = DAO.getConn();
                String sql = "select name,picture from tb_picture where id = ?";
                PreparedStatement ps = null;
                ResultSet rs = null;
                try {
                    ps = conn.prepareStatement(sql);// 创建PreparedStatement对象，并传递SQL语句
                    ps.setInt(1, id); // 为参数赋值
                    rs = ps.executeQuery();
                    if (rs.next()) {
                        String name = rs.getString(1);
                        tf_name.setText(name);
                        byte[] bt = rs.getBytes(2);
                        Icon icon = new ImageIcon(bt);
                        Dimension size = lb_picture.getSize();
                        lb_picture.setIcon(icon);
                        lb_picture.setSize(size);
                    }
                    rs.close();
                    ps.close();
                    conn.close();
                } catch (Exception e1) {
                    e1.printStackTrace();
                }
            }
        });
        button_3.setText("按编号查询");
        button_3.setBounds(211, 7, 117, 28);
        panel.add(button_3);

        final JLabel label = new JLabel();
        label.setText("编    号：");
        label.setBounds(26, 12, 66, 18);
        panel.add(label);
        //
    }
    
}
