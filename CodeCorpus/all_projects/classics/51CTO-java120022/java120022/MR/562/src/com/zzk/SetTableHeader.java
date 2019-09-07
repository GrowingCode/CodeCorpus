package com.zzk;

import java.awt.Color;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.lowagie.text.Cell;
import com.lowagie.text.Document;
import com.lowagie.text.DocumentException;
import com.lowagie.text.Element;
import com.lowagie.text.Font;
import com.lowagie.text.Paragraph;
import com.lowagie.text.Table;
import com.lowagie.text.pdf.BaseFont;
import com.lowagie.text.pdf.PdfWriter;

public class SetTableHeader {
    
    public static void main(String[] args) {
        Cell cell0 = null;// 定义单元格
        Cell cell1 = null;
        Cell cell2 = null;
        Cell cell3 = null;
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置表格的表头.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 12, Font.NORMAL);// 实例化字体
            Table table = new Table(4);// 定义表格
            cell0 = new Cell(new Paragraph("编号", FontChinese));// 创建单元格
            cell0.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置横向居中对齐
            cell0.setVerticalAlignment(Element.ALIGN_CENTER);// 设置垂直居中对齐
            cell0.setBackgroundColor(Color.GRAY);// 设置背景颜色
            cell0.setHeader(true);// 将单元格设置为表头
            cell1 = new Cell(new Paragraph("姓名", FontChinese)); // 创建单元格
            cell1.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置横向居中对齐
            cell1.setVerticalAlignment(Element.ALIGN_CENTER);// 设置垂直居中对齐
            cell1.setBackgroundColor(Color.GRAY);// 设置背景颜色
            cell1.setHeader(true);// 将单元格设置为表头
            cell2 = new Cell(new Paragraph("年龄", FontChinese)); // 创建单元格
            cell2.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置横向居中对齐
            cell2.setVerticalAlignment(Element.ALIGN_CENTER);// 设置垂直居中对齐
            cell2.setBackgroundColor(Color.GRAY);// 设置背景颜色
            cell2.setHeader(true);// 将单元格设置为表头
            cell3 = new Cell(new Paragraph("电话", FontChinese)); // 创建单元格
            cell3.setHorizontalAlignment(Element.ALIGN_CENTER);// 设置横向居中对齐
            cell3.setVerticalAlignment(Element.ALIGN_CENTER);// 设置垂直居中对齐
            cell3.setBackgroundColor(Color.GRAY);// 设置背景颜色
            cell3.setHeader(true);// 将单元格设置为表头
            // 向表格添加单元格
            table.addCell(cell0);
            table.addCell(cell1);
            table.addCell(cell2);
            table.addCell(cell3);
            table.setPadding(4);// 设置内容与单元格间距
            for (int i = 1; i <= 3; i++) {// 向表格的单元格添加内容
                table.addCell(new Paragraph("95**0" + i));
                table.addCell(new Paragraph("李*辉", FontChinese));
                table.addCell(new Paragraph("30"));
                table.addCell(new Paragraph("0431-2222****"));
            }
            document.add(table);// 将表格添加到文档
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
