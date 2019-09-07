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

public class SetCellBackgroundColor {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置单元格的背景色.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 12, Font.NORMAL);// 实例化字体
            document
                    .add(new Paragraph("                 为单元格填充颜色", FontChinese));
            Table table = new Table(5);// 定义表格
            table.addCell("1.1");// 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("2.1");
            table.addCell("3.1");
            table.addCell("2.1");
            table.addCell("2.2");
            Cell cell = new Cell();// 创建单元格
            cell.setBackgroundColor(Color.yellow);// 为单元格填充背景色
            table.addCell(cell); // 将单元格填入到表格
            table.addCell("1.3");
            table.addCell("2.3");
            Cell cell2 = new Cell();// 创建单元格
            cell2.setBackgroundColor(Color.red);// 为单元格填充背景色
            table.addCell(cell2);
            table.addCell("3.0");
            table.addCell("3.1");
            table.addCell("2.1");
            Cell cell3 = new Cell();// 创建单元格
            cell3.setBackgroundColor(Color.green);// 为单元格填充背景色
            table.addCell(cell3);
            table.addCell("3.2");
            table.addCell("1.3");
            table.addCell("2.3");
            Cell cell4 = new Cell();// 创建单元格
            cell4.setBackgroundColor(Color.red);// 为单元格填充背景色
            table.addCell(cell4); // 将单元格填入到表格
            table.addCell("2.1");
            table.addCell("2.2");
            Cell cell5 = new Cell();// 创建单元格
            cell5.setBackgroundColor(Color.blue);// 为单元格填充背景色
            table.addCell(cell5); // 将单元格填入到表格
            table.addCell("1.3");
            table.addCell("2.3");
            table.addCell("3.3");
            table.addCell("3.0");
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
