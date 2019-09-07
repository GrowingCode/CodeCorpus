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

public class SetCellColumn {
    
    public static void main(String[] args) {
        
        try {
            Document document = new Document();// 创建文档对象
            PdfWriter.getInstance(document, new FileOutputStream(
                    "C:\\设置单元格所占的列数.pdf"));// 关联文档对象与输出流
            document.open();// 打开文档
            BaseFont Chinese = BaseFont.createFont("STSong-Light",
                    "UniGB-UCS2-H", BaseFont.NOT_EMBEDDED);// 定义基础字体
            Font FontChinese = new Font(Chinese, 10, Font.NORMAL, Color.red);// 实例化字体
            Table table = new Table(5);// 定义表格
            table.addCell("1,1");// 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("2,1");
            Cell cell = new Cell(new Paragraph("我占据2列", FontChinese));// 定义一个表格单元
            cell.setColspan(2);// 设置表格列跨度(合并两个单元格)
            table.addCell(cell); // 将单元加入到表格
            table.addCell("2,1");// 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("2,2");
            table.addCell("3,2");
            table.addCell("1,3");
            table.addCell("2,3");
            table.addCell("3,3");
            Cell cell2 = new Cell(new Paragraph("我占据4列 ", FontChinese));// 定义一个表格单元
            cell2.setColspan(4);// 设置表格列跨度(合并4个单元格)
            table.addCell(cell2); // 将单元加入到表格
            table.addCell("3,1"); // 将单元格顺次的加入到表格，当一行充满时自动换行
            table.addCell("2,1");
            table.addCell("2,2");
            table.addCell("3,2");
            table.addCell("1,3");
            table.addCell("3,3");
            table.addCell("1,3");
            Cell cell3 = new Cell(new Paragraph("我占据3列 ", FontChinese));// 定义一个表格单元
            cell3.setColspan(3);// 设置表格列跨度(合并3个单元格)
            table.addCell(cell3); // 将单元加入到表格
            table.addCell("2,3");
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
