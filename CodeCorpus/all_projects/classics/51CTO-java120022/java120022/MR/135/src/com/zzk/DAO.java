package com.zzk;

import java.sql.*;
import javax.swing.JOptionPane;

public class DAO {
    private static DAO dao = new DAO(); // 声明DAO类的静态实例
    
    /**
     * 构造方法，加载数据库驱动
     */
    public DAO() {
        try {
            Class.forName("com.mysql.jdbc.Driver"); // 加载数据库驱动
        } catch (ClassNotFoundException e) {
            JOptionPane.showMessageDialog(null,
                    "数据库驱动加载失败，请将MySQL驱动配置到构建路径中。\n" + e.getMessage());
        }
    }
    
    /**
     * 获得数据库连接的方法
     * @return Connection
     */
    public static Connection getConn() {
        try {
            Connection conn = null; // 定义数据库连接
            String url = "jdbc:mysql://localhost:3306/db_database?useUnicode=true&characterEncoding=utf-8"; // 数据库db_database的URL
            String username = "root"; // 数据库的用户名
            String password = "111"; // 数据库密码
            conn = DriverManager.getConnection(url, username, password); // 建立连接
            return conn; // 返回连接
        } catch (Exception e) {
            JOptionPane.showMessageDialog(null, "数据库连接失败。" + e.getMessage());
            return null;
        }
    }
}