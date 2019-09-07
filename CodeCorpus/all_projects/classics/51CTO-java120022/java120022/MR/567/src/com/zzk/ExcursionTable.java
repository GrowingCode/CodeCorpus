package com.zzk;

import java.awt.Color;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.lowagie.text.Document;
import com.lowagie.text.DocumentException;
import com.lowagie.text.Font;
import com.lowagie.text.Paragraph;
import com.lowagie.text.Table;
import com.lowagie.text.pdf.BaseFont;
import com.lowagie.text.pdf.PdfWriter;

public class ExcursionTable {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\偏移表格.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 10, Font.NORMAL);// 实例化字体
            Table table = new Table(3);// 创建表格
            table.setBorderWidth(1);// 设置表格边框宽度
            table.setBorderColor(Color.blue);// 设置表格边框颜色
            table.setSpacing(5);// 设置表格与单元格的间距
            table.setPadding(5);// 设置单元格与内容的间距
            table.addCell("1.1");// 添加单元格
            table.addCell("1.2");
            table.addCell("1.3");
            document
                    .add(new Paragraph("                     原表格。", FontChinese));// 向文档添加内容
            document.add(table);// 向文档添加表格
            document.add(new Paragraph("                     默认的距离。",
                    FontChinese));
            document.add(table);
            document.add(new Paragraph("                     设置表格偏移值为0的距离。",
                    FontChinese));
            table.setOffset(0);// 设置表格偏移数值
            document.add(table);
            document.add(new Paragraph("                     设置表格偏移值为-15的距离。",
                    FontChinese));
            table.setOffset(-15);// 设置表格偏移数值
            document.add(table);
            document.close();// 关闭文档
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (DocumentException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
