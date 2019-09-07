package com.zzk;

import java.awt.Color;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import com.lowagie.text.Cell;
import com.lowagie.text.Document;
import com.lowagie.text.DocumentException;
import com.lowagie.text.Font;
import com.lowagie.text.Paragraph;
import com.lowagie.text.Table;
import com.lowagie.text.pdf.BaseFont;
import com.lowagie.text.pdf.PdfWriter;

public class InsertTable {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document,
                    new FileOutputStream("C:\\嵌套表格.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 10, Font.BOLDITALIC,
                    Color.BLUE);// 实例化字体
            Font font = new Font(Chinese, 10, Font.NORMAL);
            Table table1 = new Table(3);// 创建表格
            Cell cell = new Cell(new Paragraph("嵌入的表一", FontChinese));// 创建单元格
            cell.setColspan(3);// 设置列跨度
            // 单元格添入到表格行满自动换行
            table1.addCell(cell);
            table1.addCell(new Paragraph("表一 0.0", FontChinese));
            table1.addCell(new Paragraph("表一 0.1", FontChinese));
            table1.addCell(new Paragraph("表一 0.2", FontChinese));
            table1.addCell(new Paragraph("表一 1.0", FontChinese));
            table1.addCell(new Paragraph("表一 1.1", FontChinese));
            table1.addCell(new Paragraph("表一 1.2", FontChinese));
            Table table2 = new Table(2);// 创建表格
            // 单元格添入到表格，行满自动换行
            table2.addCell(new Paragraph("表二 0.0", FontChinese));
            table2.addCell(new Paragraph("表二0.1", FontChinese));
            table2.addCell(new Paragraph("表二 1.0", FontChinese));
            table2.addCell(new Paragraph("表二 1.1", FontChinese));
            Cell tableCell = new Cell(new Paragraph("使用Cell嵌入的表二", FontChinese));// 创建一个单元格
            tableCell.add(table2);// 将表格添加到单元格
            Table table3 = new Table(5, 5);// 创建5行5列的原表
            table3.insertTable(table1); // 将第一个表格嵌入到原表中第一列
            // 单元格添入到表格行满自动换行
            table3.addCell(new Paragraph("原表1.1", font));
            table3.addCell(new Paragraph("原表1.2", font));
            table3.addCell(new Paragraph("原表1.3", font));
            table3.setPadding(5);// 设置填充值为5
            table3.addCell(tableCell);// 添加单元格，实现第二个表格的嵌入
            document.add(table3);// 向文档中添加原表
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
