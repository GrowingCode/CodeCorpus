package com.zzk;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;

@SuppressWarnings("serial")
public class ConversionPictureFormatFrame extends JFrame {
    
    private JComboBox comboBox;
    private JTextField tf_pathAndFileName;
    private File imgFile = null;
    private BufferedImage buffImage;
    private DrawImagePanel imagePanel = null;
    private String path = null;
    private String fileName = null;
    private String pathAndFileName = null;
    /**
     * Launch the application
     * @param args
     */
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {
            public void run() {
                try {
                    ConversionPictureFormatFrame frame = new ConversionPictureFormatFrame();
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
    public ConversionPictureFormatFrame() {
        super();
        setTitle("转换图片格式");
        setBounds(100, 100, 432, 315);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        final JPanel panel = new JPanel();
        getContentPane().add(panel, BorderLayout.NORTH);
        
        imagePanel = new DrawImagePanel();
        getContentPane().add(imagePanel, BorderLayout.CENTER);

        final JLabel label = new JLabel();
        label.setText("请选择原图片：");
        panel.add(label);

        tf_pathAndFileName = new JTextField();
        tf_pathAndFileName.setPreferredSize(new Dimension(200,25));
        panel.add(tf_pathAndFileName);
        
        final JButton button = new JButton();
        button.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                JFileChooser fileChooser = new JFileChooser();// 创建文件选择器
                FileFilter filter = new FileNameExtensionFilter(
                        "图像文件（JPG/GIF/png/bmp)", "JPG", "JPEG", "GIF", "png", "bmp");// 创建过滤器
                fileChooser.setFileFilter(filter);// 设置过滤器
                int i = fileChooser.showOpenDialog(null);// 显示打开对话框
                if (i == JFileChooser.APPROVE_OPTION) {
                    imgFile = fileChooser.getSelectedFile(); // 获取选中图片的File对象
                }
                if (imgFile != null) {
                    try {
                        buffImage = ImageIO.read(imgFile);// 构造BufferedImage对象
                        path = imgFile.getParent();
                        fileName = imgFile.getName();
                        pathAndFileName = path + "\\" + fileName;
                        tf_pathAndFileName.setText(pathAndFileName);
                    } catch (IOException ex) {
                        ex.printStackTrace();
                    }
                }
                imagePanel.repaint();// 调用paint()方法
            }
        });
        button.setText("选择图片");
        panel.add(button);

        final JPanel panel_1 = new JPanel();
        getContentPane().add(panel_1, BorderLayout.SOUTH);

        final JLabel label_1 = new JLabel();
        label_1.setText("请选择转换后的图片格式：");
        panel_1.add(label_1);

        comboBox = new JComboBox();
        comboBox.setPreferredSize(new Dimension(130,25));
        comboBox.setModel(new DefaultComboBoxModel(new String[] {"jpg", "gif", "png", "bmp"}));
        panel_1.add(comboBox);

        final JButton button_1 = new JButton();
        button_1.addActionListener(new ActionListener() {
            public void actionPerformed(final ActionEvent e) {
                try {
                    String extName = (String)comboBox.getSelectedItem();// 新格式的扩展名，不含点
                    String pathAndName = pathAndFileName.substring(0, pathAndFileName.lastIndexOf(".") + 1)+extName;// 转换后的图片完整路径和文件名
                    File file = new File(pathAndName);// 创建转换后图片的File对象
                    ImageIO.write(buffImage, extName, file);// 将缓冲图像保存到磁盘
                    File oldFile = new File(pathAndFileName);// 原图片的File对象
                    oldFile.delete();// 删除原图片文件
                    JOptionPane.showMessageDialog(null, "文件格式更改成功！");// 显示提示信息
                } catch (IOException e1) {
                    JOptionPane.showMessageDialog(null, "保存失败\n" + e1.getMessage());// 显示提示信息
                }
            }
        });
        button_1.setText("保    存");
        panel_1.add(button_1);
        
    }
    // 创建面板类
    class DrawImagePanel extends JPanel {
        public void paint(Graphics g) {
            g.clearRect(0, 0, getWidth(), getHeight());
            g.drawImage(buffImage, 0, 0, this); // 绘制指定的图片
        }
    }
}
